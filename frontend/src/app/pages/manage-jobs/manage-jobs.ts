import { Component, OnInit, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, CustomJob } from '../../core/services/api';
import { AuthService } from '../../core/services/auth';

type ViewMode  = 'grid' | 'list';
type ModalMode = 'create' | 'edit' | null;
type Tab       = 'jobs' | 'applications';

export interface Application {
  id:               number;
  job_title:        string;
  job_location:     string;
  salary_usd:       number;
  cover_letter:     string;
  status:           string;   // 'envoyée' | 'acceptée' | 'rejetée'
  applied_at:       string;
  candidate_id:     number;   
  candidate_name:   string;
  candidate_email:  string;
  experience_level: string;
  skills:           string;
  cv_text:          string;
  cv_filename:   string;
}

interface AppGroup {
  jobTitle: string;
  apps:     Application[];
}

@Component({
  selector:    'app-manage-jobs',
  standalone:  true,
  imports:     [CommonModule, FormsModule],
  templateUrl: './manage-jobs.html',
  styleUrls:   ['./manage-jobs.css']
})
export class ManageJobsComponent implements OnInit {

  constructor(private api: ApiService, private auth: AuthService) {}

  // ── Offres ─────────────────────────────────────────────────
  jobs        = signal<CustomJob[]>([]);
  loading     = signal(true);
  saving      = signal(false);
  deletingId  = signal<number | null>(null);
  searchQuery = signal('');
  viewMode    = signal<ViewMode>('grid');
  modalMode   = signal<ModalMode>(null);
  toast       = signal<{ msg: string; type: 'success' | 'error' } | null>(null);
  form: CustomJob = this.emptyForm();

  // ── Tabs ───────────────────────────────────────────────────
  activeTab = signal<Tab>('jobs');

  // ── Candidatures ───────────────────────────────────────────
  allApplications  = signal<Application[]>([]);
  loadingApps      = signal(false);
  processingId     = signal<number | null>(null);
  appStatusFilter  = signal<string>('all');

  statusFilters = [
    { value: 'all',      label: 'Toutes'   },
    { value: 'envoyée',  label: 'En attente' },
    { value: 'acceptée', label: 'Acceptées' },
    { value: 'rejetée',  label: 'Rejetées'  },
  ];

  // Modal candidatures d'une offre spécifique
  jobAppModal: { open: boolean; jobTitle: string; apps: Application[] } =
    { open: false, jobTitle: '', apps: [] };

  // ── Computed ───────────────────────────────────────────────
  filtered = computed(() => {
    const q = this.searchQuery().toLowerCase();
    if (!q) return this.jobs();
    return this.jobs().filter(j =>
      j.job_title.toLowerCase().includes(q) ||
      (j.location || '').toLowerCase().includes(q) ||
      (j.experience_level || '').toLowerCase().includes(q)
    );
  });

  stats = computed(() => {
    const all      = this.jobs();
    const apps     = this.allApplications();
    const pending  = apps.filter(a => a.status === 'envoyée').length;
    return {
      total:       all.length,
      remote:      all.filter(j => (j.remote_ratio ?? 0) >= 100).length,
      avg:         all.length
        ? Math.round(all.reduce((s, j) => s + (j.salary_usd ?? 0), 0) / all.length)
        : 0,
      pendingApps: pending,
    };
  });

  filteredGroups = computed<AppGroup[]>(() => {
    const filter = this.appStatusFilter();
    let apps = this.allApplications();
    if (filter !== 'all') apps = apps.filter(a => a.status === filter);

    // Grouper par titre d'offre
    const map = new Map<string, Application[]>();
    apps.forEach(a => {
      if (!map.has(a.job_title)) map.set(a.job_title, []);
      map.get(a.job_title)!.push(a);
    });
    return Array.from(map.entries()).map(([jobTitle, apps]) => ({ jobTitle, apps }));
  });

  // ── Lifecycle ──────────────────────────────────────────────
  ngOnInit() {
    this.loadJobs();
    this.loadApplications();
  }

  // ── Offres CRUD ────────────────────────────────────────────
  loadJobs() {
    this.loading.set(true);
    this.api.getCustomJobs().subscribe({
      next:  jobs => { this.jobs.set(jobs); this.loading.set(false); },
      error: ()   => { this.loading.set(false); this.showToast('Erreur chargement', 'error'); }
    });
  }

  openCreate() { this.form = this.emptyForm(); this.modalMode.set('create'); }
  openEdit(job: CustomJob) { this.form = { ...job }; this.modalMode.set('edit'); }
  closeModal() { this.modalMode.set(null); }

  save() {
    if (!this.form.job_title?.trim()) return;
    this.saving.set(true);
    const obs = this.modalMode() === 'edit' && this.form.id
      ? this.api.updateJob(this.form.id, this.form)
      : this.api.createJob(this.form);
    obs.subscribe({
      next: () => {
        this.saving.set(false);
        this.closeModal();
        this.loadJobs();
        this.showToast(this.modalMode() === 'edit' ? 'Offre mise à jour ✓' : 'Offre créée ✓', 'success');
      },
      error: () => { this.saving.set(false); this.showToast('Erreur sauvegarde', 'error'); }
    });
  }

  confirmDelete(job: CustomJob) {
    if (!job.id || !confirm(`Supprimer "${job.job_title}" ?`)) return;
    this.deletingId.set(job.id);
    this.api.deleteJob(job.id).subscribe({
      next: () => {
        this.jobs.update(list => list.filter(j => j.id !== job.id));
        this.deletingId.set(null);
        this.showToast('Offre supprimée', 'success');
      },
      error: () => { this.deletingId.set(null); this.showToast('Erreur suppression', 'error'); }
    });
  }

  // ── Candidatures ───────────────────────────────────────────
  loadApplications() {
    this.loadingApps.set(true);
    this.api.getRecruiterApplications().subscribe({
      next:  apps => { this.allApplications.set(apps); this.loadingApps.set(false); },
      error: ()   => { this.loadingApps.set(false); }
    });
  }

  switchToApplications() {
    this.activeTab.set('applications');
    this.loadApplications(); // refresh à chaque visite
  }

  // Nombre de candidatures pour une offre donnée
  getAppCount(jobTitle: string): number {
    return this.allApplications().filter(a => a.job_title === jobTitle).length;
  }

  // Ouvrir le modal candidatures depuis la card offre
  openJobApplications(job: CustomJob) {
    const apps = this.allApplications().filter(a => a.job_title === job.job_title);
    this.jobAppModal = { open: true, jobTitle: job.job_title!, apps };
  }

  closeJobAppModal() {
    this.jobAppModal = { open: false, jobTitle: '', apps: [] };
  }

  // Accepter / Rejeter
  updateStatus(app: Application, status: 'acceptée' | 'rejetée') {
    this.processingId.set(app.id);
    this.api.updateApplicationStatus(app.id, status).subscribe({
      next: () => {
        this.processingId.set(null);
        // Mettre à jour localement
        this.allApplications.update(list =>
          list.map(a => a.id === app.id ? { ...a, status } : a)
        );
        // Sync modal si ouvert
        if (this.jobAppModal.open) {
          this.jobAppModal.apps = this.jobAppModal.apps.map(a =>
            a.id === app.id ? { ...a, status } : a
          );
        }
        const msg = status === 'acceptée'
          ? '✓ Candidature acceptée — email envoyé'
          : 'Candidature rejetée';
        this.showToast(msg, 'success');
      },
      error: () => {
        this.processingId.set(null);
        this.showToast('Erreur lors de la mise à jour', 'error');
      }
    });
  }

  getCvUrl(applicationId: number): string {
    return `http://localhost:8000/auth/profile/cv/application/${applicationId}?token=${this.auth.token()}`;
  }

  // ── Helpers ────────────────────────────────────────────────
  countByStatus(status: string): number {
    const apps = this.allApplications();
    if (status === 'all') return apps.length;
    return apps.filter(a => a.status === status).length;
  }

  getInitials(name: string): string {
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  }

  private emptyForm(): CustomJob {
    return { job_title: '', description: '', skills_desc: '', experience_level: 'mid', location: '', salary_usd: 0, remote_ratio: 0 };
  }

  private showToast(msg: string, type: 'success' | 'error') {
    this.toast.set({ msg, type });
    setTimeout(() => this.toast.set(null), 3500);
  }

  remoteLabel(ratio: number | undefined): string {
    if (!ratio || ratio === 0) return 'Présentiel';
    if (ratio >= 100)          return 'Full Remote';
    return 'Hybride';
  }

  levelLabel(level: string | undefined): string {
    const map: Record<string, string> = { entry: 'Junior', mid: 'Confirmé', senior: 'Senior', executive: 'Executive' };
    return map[level ?? ''] ?? level ?? '—';
  }

  formatSalary(n: number | undefined): string {
    if (!n) return '—';
    return new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n);
  }

  trackById(_: number, j: CustomJob) { return j.id; }
}