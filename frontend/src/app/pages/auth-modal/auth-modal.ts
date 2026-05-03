import { Component, Output, EventEmitter, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../core/services/auth';

@Component({
  selector: 'app-auth-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './auth-modal.html',
  styleUrls: ['./auth-modal.css']
})
export class AuthModalComponent {
  @Output() closed = new EventEmitter<void>();

  mode: 'login' | 'signup' = 'login';
  signupStep: 'form' | 'verify' = 'form';

  loading = signal(false);
  error = signal('');
  success = signal('');

  // Afficher/masquer mot de passe
  showPassword = false;
  showConfirmPassword = false;

  form = {
    email: '',
    password: '',
    confirm_password: '',
    full_name: ''
  };

  verifyCode = '';

  constructor(private auth: AuthService) {}

  submit() {
    this.error.set('');
    this.loading.set(true);

    if (this.mode === 'login') {
      this.auth.login(this.form.email, this.form.password).subscribe({
        next: () => { this.loading.set(false); this.closed.emit(); },
        error: (e) => { this.error.set(e.error?.detail || 'Erreur'); this.loading.set(false); }
      });
    } else {
      // Étape 1 : envoyer le code
      if (this.form.password !== this.form.confirm_password) {
        this.error.set('Les mots de passe ne correspondent pas');
        this.loading.set(false);
        return;
      }
      this.auth.sendVerificationCode(
        this.form.email, this.form.password,
        this.form.confirm_password, this.form.full_name
      ).subscribe({
        next: () => {
          this.loading.set(false);
          this.signupStep = 'verify';
          this.success.set(`Code envoyé à ${this.form.email}`);
        },
        error: (e) => { this.error.set(e.error?.detail || 'Erreur'); this.loading.set(false); }
      });
    }
  }

  verifyAndCreate() {
    this.error.set('');
    this.loading.set(true);
    this.auth.verifyCode(this.form.email, this.verifyCode).subscribe({
      next: () => { this.loading.set(false); this.closed.emit(); },
      error: (e) => { this.error.set(e.error?.detail || 'Code incorrect'); this.loading.set(false); }
    });
  }

  resendCode() {
    this.error.set('');
    this.success.set('');
    this.loading.set(true);
    this.auth.sendVerificationCode(
      this.form.email, this.form.password,
      this.form.confirm_password, this.form.full_name
    ).subscribe({
      next: () => { this.loading.set(false); this.success.set('Nouveau code envoyé !'); },
      error: (e) => { this.error.set(e.error?.detail || 'Erreur'); this.loading.set(false); }
    });
  }

  switchMode() {
    this.mode = this.mode === 'login' ? 'signup' : 'login';
    this.signupStep = 'form';
    this.error.set('');
    this.success.set('');
    this.form = { email: '', password: '', confirm_password: '', full_name: '' };
    this.verifyCode = '';
  }

  close() { this.closed.emit(); }
}