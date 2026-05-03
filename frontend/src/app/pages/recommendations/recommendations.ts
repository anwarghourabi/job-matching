import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AuthService } from '../../core/services/auth';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-recommendations',
  standalone: true,
  imports: [CommonModule ,RouterLink, RouterLinkActive],
  templateUrl: './recommendations.html',
  styleUrls: ['./recommendations.css']
})
export class RecommendationsComponent implements OnInit {
  public auth = inject(AuthService);
  private router = inject(Router);
  menuOpen = signal(false);

  recommendations = signal<any[]>([]);
  history = signal<any[]>([]);
  loading = signal(true);
  basedOn = signal<any>({ cv: false, history: 0 });

  ngOnInit() {
    if (!this.auth.isLoggedIn()) {
      this.router.navigate(['/']);
      return;
    }
    // Petit délai pour éviter le conflit de navigation
    setTimeout(() => this.load(), 0);
  }

  load() {
    this.loading.set(true);
    this.auth.getRecommendations().subscribe({
      next: (res) => {
        this.recommendations.set(res.recommendations);
        this.basedOn.set(res.based_on);
        this.loading.set(false);
      },
      error: () => this.loading.set(false)
    });
    this.auth.getHistory().subscribe(h => this.history.set(h));
  }

  remoteLabel(ratio: number): string {
    if (ratio === 100) return '🌍 Remote';
    if (ratio >= 50) return '🏠 Hybride';
    return '🏢 Présentiel';
  }
    closeMenu() { this.menuOpen.set(false); }

}