import { Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { ApiService } from '../../../core/services/api'; 
import { AuthModalComponent } from '../../../pages/auth-modal/auth-modal';
import { AuthService } from '../../../core/services/auth';
@Component({
  selector: 'app-header',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, AuthModalComponent],
  templateUrl: './header.html',
  styleUrls: ['./header.css']
})
export class HeaderComponent implements OnInit {
  private api = inject(ApiService);
  engineReady = signal(false);
  menuOpen = signal(false);
  showAuthModal = false;
  
  constructor(public auth: AuthService) {}

  ngOnInit() {
    this.api.health().subscribe({
      next: h => this.engineReady.set(h.engine_ready),
      error: () => this.engineReady.set(false)
    });
  }

  toggleMenu() { this.menuOpen.update(v => !v); }
  closeMenu() { this.menuOpen.set(false); }

  
}