import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api';
import { StatsResponse } from '../../core/models/api.models';

@Component({
  selector: 'app-stats',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './stats.html',
  styleUrls: ['./stats.css']
})
export class StatsComponent implements OnInit {
  private api = inject(ApiService);

  stats = signal<StatsResponse | null>(null);
  loading = signal(true);
  error = signal('');

  ngOnInit() {
    this.api.stats().subscribe({
      next: s => { this.stats.set(s); this.loading.set(false); },
      error: e => { this.error.set(e.error?.detail || 'Erreur API'); this.loading.set(false); }
    });
  }

  entries(obj: Record<string, number>): [string, number][] {
    return Object.entries(obj).sort((a, b) => b[1] - a[1]);
  }

  maxVal(obj: Record<string, number>): number {
    return Math.max(...Object.values(obj));
  }

  barWidth(val: number, max: number): string {
    return `${Math.round((val / max) * 100)}%`;
  }
}