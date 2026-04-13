import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MatchFile } from './match-file';

describe('MatchFile', () => {
  let component: MatchFile;
  let fixture: ComponentFixture<MatchFile>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MatchFile]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MatchFile);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
