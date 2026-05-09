import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api';
import { AuthService } from '../../core/services/auth';
import { JobCrudModal } from '../job-crud-modal/job-crud-modal';
import { AuthModalComponent } from '../auth-modal/auth-modal';
import { Observable, of } from 'rxjs';

@Component({
  selector: 'app-jobs',
  standalone: true,
  imports: [CommonModule, FormsModule, JobCrudModal, AuthModalComponent],
  templateUrl: './jobs.html',
  styleUrls: ['./jobs.css']
})
export class JobsComponent implements OnInit {
  private api  = inject(ApiService);
  auth = inject(AuthService);
  selectedCv = signal<File | null>(null);

  // ── Liste offres ───────────────────────────────────────────
  jobs          = signal<any[]>([]);
  total         = signal(0);
  pages         = signal(0);
  page          = signal(1);
  loading       = signal(false);
  error         = signal('');
  searchQuery   = '';
  searchResults = signal<any[] | null>(null);
  searchCount   = signal(0);
  searching     = signal(false);
  filters       = { source: '', level: '' };
  perPage       = 20;

  // ── Modals ─────────────────────────────────────────────────
  showCrudModal = false;
  showAuthModal = false;

  // ── Candidatures ───────────────────────────────────────────
  appliedTitles = signal<Set<string>>(new Set()); // titres déjà postulés
  applying      = signal(false);
  applyError    = signal('');
  toast         = signal<{ msg: string; type: 'success' | 'error' } | null>(null);

  applyModal: {
    open:        boolean;
    job:         any;
    coverLetter: string;
  } = { open: false, job: null, coverLetter: '' };

  // ── Init ───────────────────────────────────────────────────
  ngOnInit() {
    this.loadJobs();
    if (this.auth.isLoggedIn()) {
      this.loadApplied();
    }
  }

  // ── Chargement offres ──────────────────────────────────────
  loadJobs() {
    this.loading.set(true);
    this.searchResults.set(null);
    this.api.listJobs(this.page(), this.perPage, this.filters.source, this.filters.level).subscribe({
      next: r => {
        this.jobs.set(r.jobs);
        this.total.set(r.total);
        this.pages.set(r.pages);
        this.loading.set(false);
      },
      error: e => { this.error.set(e.error?.detail || 'Erreur API'); this.loading.set(false); }
    });
  }

  // ── Charger candidatures existantes ───────────────────────
  loadApplied() {
    this.api.getApplications().subscribe({
      next: (apps: any[]) => {
        this.appliedTitles.set(new Set(apps.map(a => a.job_title)));
      },
      error: () => {}
    });
  }

  isApplied(jobTitle: string): boolean {
    return this.appliedTitles().has(jobTitle);
  }

  // ── Ouvrir modal postuler ──────────────────────────────────
  openApply(job: any) {
    this.applyError.set('');
    this.selectedCv.set(null);
    this.applyModal = { open: true, job, coverLetter: '' };
  }

  closeApply() {
    this.applyModal = { open: false, job: null, coverLetter: '' };
    this.applyError.set('');
    this.selectedCv.set(null);
  }

  // ── Soumettre candidature ──────────────────────────────────
submitApply() {
  if (!this.auth.isLoggedIn()) return;
  this.applying.set(true);
  this.applyError.set('');

  const job = this.applyModal.job;
  const cv  = this.selectedCv();

  // Si un nouveau CV est sélectionné, l'uploader d'abord
  const upload$ = cv ? this.auth.uploadCv(cv) : of(null);


  upload$.subscribe({
    next: () => {
      this.auth.applyToJob({
        job_title:    job.job_title,
        job_source:   job.source   || '',
        job_location: job.location || '',
        salary_usd:   job.salary_usd || 0,
        cover_letter: this.applyModal.coverLetter
      }).subscribe({
        next: () => {
          this.applying.set(false);
          this.appliedTitles.update(s => {
            const next = new Set(s);
            next.add(job.job_title);
            return next;
          });
          this.auth.addToHistory({
            job_title:  job.job_title,
            job_source: job.source || '',
            score:      0
          }).subscribe();
          this.closeApply();
          this.showToast('Candidature envoyée avec succès ! 🚀', 'success');
        },
        error: (e) => {
          this.applying.set(false);
          if (e.status === 409) {
            this.appliedTitles.update(s => { const n = new Set(s); n.add(job.job_title); return n; });
            this.closeApply();
          } else {
            this.applyError.set(e.error?.detail || 'Erreur lors de l\'envoi');
          }
        }
      });
    },
    error: () => {
      this.applying.set(false);
      this.applyError.set('Erreur upload CV');
    }
  });
}

  // ── Auth modal ─────────────────────────────────────────────
  onAuthClosed() {
    this.showAuthModal = false;
    if (this.auth.isLoggedIn()) {
      this.loadApplied();
      // Ré-ouvrir le modal si un job était en attente
      if (this.applyModal.job) {
        this.applyModal.open = true;
      }
    }
  }

  // ── CRUD modal ─────────────────────────────────────────────
  onModalClosed() {
    this.showCrudModal = false;
    this.loadJobs();
  }

  // ── Recherche & filtres ────────────────────────────────────
  search() {
    if (!this.searchQuery.trim()) { this.searchResults.set(null); return; }
    this.searching.set(true);
    this.api.searchJobs(this.searchQuery, 50).subscribe({
      next: r => {
        this.searchResults.set(r.results);
        this.searchCount.set(r.count);
        this.searching.set(false);
      },
      error: e => { this.error.set(e.error?.detail || 'Erreur API'); this.searching.set(false); }
    });
  }

  clearSearch() {
    this.searchQuery = '';
    this.searchResults.set(null);
    this.searchCount.set(0);
  }

  applyFilters() { this.page.set(1); this.loadJobs(); }

  goTo(p: number) {
    if (p < 1 || p > this.pages()) return;
    this.page.set(p);
    this.loadJobs();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  get pageNumbers(): number[] {
    const total = this.pages(), current = this.page(), delta = 2;
    const range: number[] = [];
    for (let i = Math.max(1, current - delta); i <= Math.min(total, current + delta); i++) range.push(i);
    return range;
  }

  get displayedJobs(): any[] { return this.searchResults() ?? this.jobs(); }

  remoteLabel(ratio: number): string {
    if (ratio === 100) return '🌍 Remote';
    if (ratio >= 50)  return '🏠 Hybride';
    return '🏢 Présentiel';
  }

  private showToast(msg: string, type: 'success' | 'error') {
    this.toast.set({ msg, type });
    setTimeout(() => this.toast.set(null), 3500);
  }

  onCvChange(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const allowed = ['.pdf', '.docx', '.doc', '.txt'];
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  if (!allowed.includes(ext)) {
    this.applyError.set('Format non supporté : PDF, DOCX, DOC ou TXT');
    return;
  }
  this.selectedCv.set(file);
}

removeCv() { this.selectedCv.set(null); }

}