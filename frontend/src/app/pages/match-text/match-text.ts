import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api';
import { AuthService } from '../../core/services/auth';
import { MatchRequest, MatchResponse } from '../../core/models/api.models';
import { JobCardComponent } from '../../shared/components/job-card/job-card';

@Component({
  selector: 'app-match-text',
  standalone: true,
  imports: [CommonModule, FormsModule, JobCardComponent],
  templateUrl: './match-text.html',
  styleUrls: ['./match-text.css']
})
export class MatchTextComponent {
  private api = inject(ApiService);
  private auth = inject(AuthService);

  form: MatchRequest = {
    name: '',
    cv_text: '',
    experience_level: 'auto',
    desired_location: '',
    min_salary: 0,
    max_salary: 0,
    remote_only: false,
    employment_type: '',
    top_k: 10
  };

  loading = signal(false);
  result = signal<MatchResponse | null>(null);
  error = signal<string>('');

  examples = [
    {
      label: ' Dev Angular',
      text: 'Développeur Angular 5 ans expérience, TypeScript, RxJS, NgRx, REST APIs, Docker, CI/CD. Certifié AWS.'
    },
    {
      label: ' Comptable',
      text: 'Comptable confirmée, 5 ans. Comptabilité générale, Audit, IFRS, Sage, Excel, Contrôle de gestion.'
    },
    {
      label: ' Marketing',
      text: 'Chef de projet digital, SEO/SEM, Google Ads, Analytics, CRM Salesforce, gestion réseaux sociaux.'
    },
    {
      label: ' RH',
      text: 'Responsable RH, recrutement, formation, GPEC, paie, droit du travail, ADP, 8 ans expérience.'
    }
  ];

  loadExample(text: string) {
    this.form.cv_text = text;
  }

  submit() {
  if (!this.form.cv_text.trim()) return;
  this.loading.set(true);
  this.error.set('');
  this.result.set(null);

  // ── Forcer les types numériques ──
  const payload: MatchRequest = {
    ...this.form,
    top_k: Number(this.form.top_k),
    min_salary: Number(this.form.min_salary),
    max_salary: Number(this.form.max_salary),
  };

  this.api.matchText(payload).subscribe({
    next: r => {
      this.result.set(r);
      this.loading.set(false);

      if (this.auth.isLoggedIn()) {
        this.auth.updateProfile({
          cv_text: this.form.cv_text,
          experience_level: this.form.experience_level !== 'auto'
            ? this.form.experience_level
            : r.experience_level,
          desired_location: this.form.desired_location || undefined,
          skills: r.skills_detected.join(', '),
        }).subscribe();

        r.results.slice(0, 5).forEach(job => {
          this.auth.addToHistory({
            job_title: job.job_title,
            job_source: job.source,
            score: job.final_score
          }).subscribe();
        });
      }
    },
    error: e => {
      this.error.set(e.error?.detail || 'Erreur de connexion à l\'API');
      this.loading.set(false);
    }
  });
} 
}