import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api';
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

    this.api.matchText(this.form).subscribe({
      next: r => { this.result.set(r); this.loading.set(false); },
      error: e => {
        this.error.set(e.error?.detail || 'Erreur de connexion à l\'API');
        this.loading.set(false);
      }
    });
  }
}