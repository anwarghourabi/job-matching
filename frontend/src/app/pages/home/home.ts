import { Component, OnInit, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api';
import { HealthResponse } from '../../core/models/api.models';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './home.html',
  styleUrls: ['./home.css']
})
export class HomeComponent implements OnInit {
  private api = inject(ApiService);
  health = signal<HealthResponse | null>(null);
  error = signal(false);

  ngOnInit() {
    this.api.health().subscribe({
      next: h => this.health.set(h),
      error: () => this.error.set(true)
    });
  }
}