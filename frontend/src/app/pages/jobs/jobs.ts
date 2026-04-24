import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api';
import { JobCrudModal } from '../job-crud-modal/job-crud-modal';

@Component({
  selector: 'app-jobs',
  standalone: true,
  imports: [CommonModule, FormsModule, JobCrudModal],
  templateUrl: './jobs.html',
  styleUrls: ['./jobs.css']
})
export class JobsComponent implements OnInit {
  private api = inject(ApiService);

  jobs = signal<any[]>([]);
  total = signal(0);
  pages = signal(0);
  page = signal(1);
  loading = signal(false);
  error = signal('');

  searchQuery = '';
  searchResults = signal<any[] | null>(null);
  searchCount = signal(0);
  searching = signal(false);

  filters = { source: '', level: '' };
  perPage = 20;

  // Modal CRUD
  showCrudModal = false;

  ngOnInit() { this.loadJobs(); }

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

  // Fermer le modal et recharger les offres
  onModalClosed() {
    this.showCrudModal = false;
    this.loadJobs();
  }

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
    const total = this.pages();
    const current = this.page();
    const delta = 2;
    const range: number[] = [];
    for (let i = Math.max(1, current - delta); i <= Math.min(total, current + delta); i++) {
      range.push(i);
    }
    return range;
  }

  get displayedJobs(): any[] {
    return this.searchResults() ?? this.jobs();
  }

  remoteLabel(ratio: number): string {
    if (ratio === 100) return '🌍 Remote';
    if (ratio >= 50) return '🏠 Hybride';
    return '🏢 Présentiel';
  }
}