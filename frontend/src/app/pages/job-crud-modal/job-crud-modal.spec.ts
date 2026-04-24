import { ComponentFixture, TestBed } from '@angular/core/testing';

import { JobCrudModal } from './job-crud-modal';

describe('JobCrudModal', () => {
  let component: JobCrudModal;
  let fixture: ComponentFixture<JobCrudModal>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [JobCrudModal]
    })
    .compileComponents();

    fixture = TestBed.createComponent(JobCrudModal);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
