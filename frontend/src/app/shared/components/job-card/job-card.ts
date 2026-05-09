import { Component, Input, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { JobResult } from '../../../core/models/api.models';
import { AuthService } from '../../../core/services/auth';
import { ApiService } from '../../../core/services/api';

@Component({
  selector: 'app-job-card',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './job-card.html',
  styleUrls: ['./job-card.css']
})
export class JobCardComponent {
  @Input() job!: JobResult;

  public auth = inject(AuthService);
  private api = inject(ApiService);

  expanded    = false;
  applyOpen   = signal(false);
  applying    = signal(false);
  applied     = signal(false);
  applyError  = signal('');
  coverLetter = '';

  toggleExpand() { this.expanded = !this.expanded; }

  // ── Postuler ───────────────────────────────────────────────
  openApply(e: Event) {
    e.stopPropagation();   // ne pas déclencher toggleExpand
    if (this.applied()) return;
    this.applyOpen.set(true);
    this.applyError.set('');
    this.coverLetter = '';
  }

  closeApply(e?: Event) {
    e?.stopPropagation();
    this.applyOpen.set(false);
    this.applyError.set('');
  }

  submitApply(e: Event) {
    e.stopPropagation();
    if (!this.auth.isLoggedIn()) {
      this.applyError.set('Connectez-vous pour postuler.');
      return;
    }
    this.applying.set(true);
    this.api.applyToJob({
      job_title:    this.job.job_title,
      job_source:   this.job.source   || '',
      job_location: this.job.location || '',
      salary_usd:   this.job.salary_usd || 0,
      cover_letter: this.coverLetter
    }).subscribe({
      next: () => {
        this.applying.set(false);
        this.applied.set(true);
        this.applyOpen.set(false);
        // Ajouter à l'historique
        this.auth.addToHistory({
          job_title:  this.job.job_title,
          job_source: this.job.source || '',
          score:      this.job.final_score
        }).subscribe();
      },
      error: (err) => {
        this.applying.set(false);
        // 409 = déjà postulé → on marque quand même comme postulé
        if (err.status === 409) {
          this.applied.set(true);
          this.applyOpen.set(false);
        } else {
          this.applyError.set(err.error?.detail || 'Erreur lors de l\'envoi');
        }
      }
    });
  }

  // ── Getters ────────────────────────────────────────────────
  get isRecruiter(): boolean {
    return this.auth.currentUser()?.role === 'recruteur';
  }

  get isLoggedIn(): boolean {
    return this.auth.isLoggedIn();
  }

  get scoreColor(): string {
    if (this.job.final_score >= 0.75) return 'excellent';
    if (this.job.final_score >= 0.55) return 'good';
    if (this.job.final_score >= 0.40) return 'fair';
    return 'low';
  }

  get scorePercent(): number {
    return Math.round(this.job.final_score * 100);
  }

  get remoteLabel(): string {
    if (this.job.remote_ratio === 100) return '🌍 Full Remote';
    if (this.job.remote_ratio >= 50)  return '🏠 Hybride';
    return '🏢 Présentiel';
  }
}