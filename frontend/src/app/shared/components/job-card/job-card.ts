import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { JobResult } from '../../../core/models/api.models'; 

@Component({
  selector: 'app-job-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './job-card.html',
  styleUrls: ['./job-card.css']
})
export class JobCardComponent {
  @Input() job!: JobResult;
  expanded = false;

  toggleExpand() { this.expanded = !this.expanded; }

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
    if (this.job.remote_ratio >= 50) return '🏠 Hybride';
    return '🏢 Présentiel';
  }
}