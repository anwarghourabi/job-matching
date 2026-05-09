import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api';
import { AuthService } from '../../core/services/auth';
import { MatchResponse } from '../../core/models/api.models';
import { JobCardComponent } from '../../shared/components/job-card/job-card';

@Component({
  selector: 'app-match-file',
  standalone: true,
  imports: [CommonModule, FormsModule, JobCardComponent],
  templateUrl: './match-file.html',
  styleUrls: ['./match-file.css']
})
export class MatchFileComponent {
  private api = inject(ApiService);
  private auth = inject(AuthService);

  selectedFile = signal<File | null>(null);
  dragOver = signal(false);

  params = {
    experience_level: 'auto',
    desired_location: '',
    min_salary: 0,
    remote_only: false,
    employment_type: '',
    top_k: 10
  };

  loading = signal(false);
  result = signal<MatchResponse | null>(null);
  error = signal<string>('');

  onDragOver(e: DragEvent) { e.preventDefault(); this.dragOver.set(true); }
  onDragLeave() { this.dragOver.set(false); }

  onDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOver.set(false);
    const file = e.dataTransfer?.files[0];
    if (file) this.setFile(file);
  }

  onFileChange(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) this.setFile(file);
  }

  setFile(file: File) {
    const allowed = ['.pdf', '.docx', '.doc', '.txt'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowed.includes(ext)) {
      this.error.set('Format non supporté. Utilise PDF, DOCX, DOC ou TXT.');
      return;
    }
    this.selectedFile.set(file);
    this.error.set('');
    this.result.set(null);
  }

  removeFile() { this.selectedFile.set(null); this.result.set(null); }

  get fileSizeLabel(): string {
    const f = this.selectedFile();
    if (!f) return '';
    const kb = f.size / 1024;
    return kb > 1024 ? `${(kb/1024).toFixed(1)} Mo` : `${kb.toFixed(0)} Ko`;
  }

  submit() {
    const file = this.selectedFile();
    if (!file) return;
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);

    this.api.matchFile(file, this.params).subscribe({
      next: r => {
        this.result.set(r);
        this.loading.set(false);

        // Sauvegarder dans le profil si connecté
        if (this.auth.isLoggedIn()) {
          this.auth.updateProfile({
            experience_level: r.experience_level,
            skills: r.skills_detected.join(', '),
            desired_location: this.params.desired_location || undefined,
          }).subscribe();

           // ← Uploader le fichier CV binaire
          this.auth.uploadCv(file).subscribe();

          // Historique top 5
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
        this.error.set(e.error?.detail || 'Erreur lors du traitement du fichier');
        this.loading.set(false);
      }
    });
  }
}