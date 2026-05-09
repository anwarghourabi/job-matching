import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { AdminService } from '../../core/services/admin';
import { ApiService } from '../../core/services/api';
import { StatsResponse } from '../../core/models/api.models';

type AdminTab = 'dashboard' | 'users' | 'jobs' | 'applications';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './admin.html',
  styleUrls: ['./admin.css']
})
export class AdminComponent implements OnInit {
  private adminSvc = inject(AdminService);
  private apiSvc   = inject(ApiService);

  activeTab = signal<AdminTab>('dashboard');
  toast     = signal<{ msg: string; type: 'success' | 'error' } | null>(null);

  // ── Dashboard ──────────────────────────────────────────────
  dashboard    = signal<any>(null);
  corpusStats  = signal<StatsResponse | null>(null);
  loadingDash  = signal(true);

  // ── Users ──────────────────────────────────────────────────
  users        = signal<any[]>([]);
  usersTotal   = signal(0);
  usersPages   = signal(1);
  usersPage    = signal(1);
  usersSearch  = '';
  usersRole    = '';
  loadingUsers = signal(false);
  deletingUserId = signal<number | null>(null);

  // ── Jobs ───────────────────────────────────────────────────
  adminJobs      = signal<any[]>([]);
  jobsTotal      = signal(0);
  jobsPages      = signal(1);
  jobsPage       = signal(1);
  jobsSearch     = '';
  loadingJobs    = signal(false);
  deletingJobId  = signal<number | null>(null);

  // ── Applications ───────────────────────────────────────────
  adminApps      = signal<any[]>([]);
  appsTotal      = signal(0);
  appsPages      = signal(1);
  appsPage       = signal(1);
  appsStatus     = '';
  loadingApps    = signal(false);

  // ── Init ───────────────────────────────────────────────────
  ngOnInit() {
    this.loadDashboard();
  }

  switchTab(tab: AdminTab) {
    this.activeTab.set(tab);
    if (tab === 'dashboard' && !this.dashboard())   this.loadDashboard();
    if (tab === 'users'     && !this.users().length) this.loadUsers();
    if (tab === 'jobs'      && !this.adminJobs().length) this.loadJobs();
    if (tab === 'applications' && !this.adminApps().length) this.loadApplications();
  }

  // ── Dashboard ──────────────────────────────────────────────
  loadDashboard() {
    this.loadingDash.set(true);
    this.adminSvc.getDashboard().subscribe({
      next: d => { this.dashboard.set(d); this.loadingDash.set(false); }
    });
    this.apiSvc.stats().subscribe({
      next: s => this.corpusStats.set(s)
    });
  }

  // ── Users ──────────────────────────────────────────────────
  loadUsers() {
    this.loadingUsers.set(true);
    this.adminSvc.getUsers(this.usersSearch, this.usersRole, this.usersPage()).subscribe({
      next: r => {
        this.users.set(r.users);
        this.usersTotal.set(r.total);
        this.usersPages.set(r.pages);
        this.loadingUsers.set(false);
      }
    });
  }

  updateRole(user: any, role: string) {
    this.adminSvc.updateUserRole(user.id, role).subscribe({
      next: () => {
        this.users.update(list => list.map(u => u.id === user.id ? { ...u, role } : u));
        this.showToast(`Rôle mis à jour : ${role}`, 'success');
      },
      error: () => this.showToast('Erreur mise à jour rôle', 'error')
    });
  }

  deleteUser(user: any) {
    if (!confirm(`Supprimer "${user.full_name}" ? Toutes ses données seront perdues.`)) return;
    this.deletingUserId.set(user.id);
    this.adminSvc.deleteUser(user.id).subscribe({
      next: () => {
        this.users.update(list => list.filter(u => u.id !== user.id));
        this.usersTotal.update(n => n - 1);
        this.deletingUserId.set(null);
        this.showToast('Utilisateur supprimé', 'success');
        this.loadDashboard();
      },
      error: () => { this.deletingUserId.set(null); this.showToast('Erreur suppression', 'error'); }
    });
  }

  usersGoTo(p: number) {
    if (p < 1 || p > this.usersPages()) return;
    this.usersPage.set(p);
    this.loadUsers();
  }

  // ── Jobs ───────────────────────────────────────────────────
  loadJobs() {
    this.loadingJobs.set(true);
    this.adminSvc.getJobs(this.jobsSearch, this.jobsPage()).subscribe({
      next: r => {
        this.adminJobs.set(r.jobs);
        this.jobsTotal.set(r.total);
        this.jobsPages.set(r.pages);
        this.loadingJobs.set(false);
      }
    });
  }

  deleteJob(job: any) {
    if (!confirm(`Supprimer l'offre "${job.job_title}" ?`)) return;
    this.deletingJobId.set(job.id);
    this.adminSvc.deleteJob(job.id).subscribe({
      next: () => {
        this.adminJobs.update(list => list.filter(j => j.id !== job.id));
        this.jobsTotal.update(n => n - 1);
        this.deletingJobId.set(null);
        this.showToast('Offre supprimée', 'success');
        this.loadDashboard();
      },
      error: () => { this.deletingJobId.set(null); this.showToast('Erreur suppression', 'error'); }
    });
  }

  jobsGoTo(p: number) {
    if (p < 1 || p > this.jobsPages()) return;
    this.jobsPage.set(p);
    this.loadJobs();
  }

  // ── Applications ───────────────────────────────────────────
  loadApplications() {
    this.loadingApps.set(true);
    this.adminSvc.getApplications(this.appsStatus, this.appsPage()).subscribe({
      next: r => {
        this.adminApps.set(r.applications);
        this.appsTotal.set(r.total);
        this.appsPages.set(r.pages);
        this.loadingApps.set(false);
      }
    });
  }

  appsGoTo(p: number) {
    if (p < 1 || p > this.appsPages()) return;
    this.appsPage.set(p);
    this.loadApplications();
  }

  // ── Helpers ────────────────────────────────────────────────
  get userPageNumbers(): number[] { return this.pageRange(this.usersPage(), this.usersPages()); }
  get jobPageNumbers():  number[] { return this.pageRange(this.jobsPage(),  this.jobsPages()); }
  get appPageNumbers():  number[] { return this.pageRange(this.appsPage(),  this.appsPages()); }

  private pageRange(current: number, total: number): number[] {
    const range: number[] = [];
    for (let i = Math.max(1, current - 2); i <= Math.min(total, current + 2); i++) range.push(i);
    return range;
  }

  entries(obj: Record<string, number>): [string, number][] {
    return Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
  }
  maxVal(obj: Record<string, number>): number {
    return Math.max(...Object.values(obj || { _: 1 }));
  }
  barWidth(val: number, max: number): string {
    return `${Math.round((val / max) * 100)}%`;
  }
  formatSalary(n: number): string {
    if (!n) return '—';
    return new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n);
  }
  remoteLabel(r: number): string {
    if (r >= 100) return 'Remote'; if (r >= 50) return 'Hybride'; return 'Présentiel';
  }
  roleLabel(role: string): string {
    const m: Record<string,string> = { candidat: 'Candidat', recruteur: 'Recruteur', admin: 'Admin' };
    return m[role] ?? role;
  }
  statusClass(s: string): string {
    return s === 'acceptée' ? 'accepted' : s === 'rejetée' ? 'rejected' : 'pending';
  }

  private showToast(msg: string, type: 'success' | 'error') {
    this.toast.set({ msg, type });
    setTimeout(() => this.toast.set(null), 3000);
  }

  get maxRegistrations(): number {
  const data = this.dashboard()?.registrations_7d ?? [];
  return data.length ? Math.max(...data.map((d: any) => d.count)) : 1;
}

get maxApplications(): number {
  const data = this.dashboard()?.applications_7d ?? [];
  return data.length ? Math.max(...data.map((d: any) => d.count)) : 1;
}
}