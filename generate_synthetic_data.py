"""
generate_synthetic_data.py
Generates a realistic synthetic student dataset for testing the
EduXAI custom CSV upload feature.

Output: synthetic_student_data.csv
Students: 500
Subjects: 4 (Mathematics, Physics, Chemistry, Computer Science)
Sessional marks: 3 per subject (out of 30 each)
Final exam: 1 per subject (out of 70)
Attendance: percentage (0-100)
Past failures: count (0-3)
Final Result: weighted average out of 100
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

# ── Student base ability (hidden variable driving everything) ────────────────
ability = np.random.normal(0.6, 0.18, N).clip(0.1, 1.0)

# ── Attendance (correlated with ability) ─────────────────────────────────────
attendance = (ability * 70 + np.random.normal(20, 8, N)).clip(40, 100).round(1)

# ── Past failures (inverse of ability) ───────────────────────────────────────
fail_prob = np.clip(1 - ability, 0, 1)
past_failures = np.random.binomial(3, fail_prob * 0.4, N)

# ── Study hours per week ──────────────────────────────────────────────────────
study_hours = (ability * 15 + np.random.normal(3, 2, N)).clip(1, 20).round(1)

# ── Helper: generate marks for one subject ────────────────────────────────────
def subject_marks(ability, subject_difficulty=1.0, noise=4):
    """Generate 3 sessional marks (out of 30) and 1 final (out of 70)."""
    base = ability / subject_difficulty

    s1 = (base * 28 + np.random.normal(0, noise, N)).clip(0, 30).round(1)
    # S2 slightly correlated with S1 — same student
    s2 = (0.6 * s1 + 0.4 * (base * 28) + np.random.normal(0, noise, N)).clip(0, 30).round(1)
    # S3 slightly better on average — student learns
    s3 = (0.5 * s2 + 0.5 * (base * 28 + 1) + np.random.normal(0, noise, N)).clip(0, 30).round(1)
    # Final exam — harder, more variance
    final = (base * 62 + np.random.normal(0, noise * 1.5, N)).clip(0, 70).round(1)

    return s1, s2, s3, final

# ── Generate marks for 4 subjects ────────────────────────────────────────────
math_s1, math_s2, math_s3, math_final       = subject_marks(ability, 1.10, 4)
physics_s1, physics_s2, physics_s3, phys_final = subject_marks(ability, 1.15, 5)
chem_s1, chem_s2, chem_s3, chem_final       = subject_marks(ability, 1.05, 4)
cs_s1, cs_s2, cs_s3, cs_final               = subject_marks(ability, 0.95, 3)

# ── Compute total per subject (out of 100) ────────────────────────────────────
math_total   = (math_s1 + math_s2 + math_s3 + math_final).round(1)
phys_total   = (physics_s1 + physics_s2 + physics_s3 + phys_final).round(1)
chem_total   = (chem_s1 + chem_s2 + chem_s3 + chem_final).round(1)
cs_total     = (cs_s1 + cs_s2 + cs_s3 + cs_final).round(1)

# ── Overall result = average of 4 subjects ───────────────────────────────────
# Attendance penalty: each 10% below 75 reduces result by 2 points
att_penalty = np.clip((75 - attendance) / 10 * 2, 0, 10)
overall = ((math_total + phys_total + chem_total + cs_total) / 4 - att_penalty).clip(0, 100).round(1)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    # Demographics
    "Student_ID":       [f"S{str(i+1).zfill(4)}" for i in range(N)],

    # Behavioral
    "Attendance_%":     attendance,
    "Study_Hours_Week": study_hours,
    "Past_Failures":    past_failures,

    # Mathematics
    "Math_Sessional1":  math_s1,
    "Math_Sessional2":  math_s2,
    "Math_Sessional3":  math_s3,
    "Math_Final_Exam":  math_final,

    # Physics
    "Physics_Sessional1": physics_s1,
    "Physics_Sessional2": physics_s2,
    "Physics_Sessional3": physics_s3,
    "Physics_Final_Exam": phys_final,

    # Chemistry
    "Chem_Sessional1":  chem_s1,
    "Chem_Sessional2":  chem_s2,
    "Chem_Sessional3":  chem_s3,
    "Chem_Final_Exam":  chem_final,

    # Computer Science
    "CS_Sessional1":    cs_s1,
    "CS_Sessional2":    cs_s2,
    "CS_Sessional3":    cs_s3,
    "CS_Final_Exam":    cs_final,

    # Results
    "Math_Total_100":   math_total,
    "Physics_Total_100":phys_total,
    "Chem_Total_100":   chem_total,
    "CS_Total_100":     cs_total,
    "Overall_Result":   overall,
})

df.to_csv("/mnt/user-data/outputs/synthetic_student_data.csv", index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 55)
print("   Synthetic Student Dataset — Generation Complete")
print("=" * 55)
print(f"  Total students    : {N}")
print(f"  Subjects          : 4 (Math, Physics, Chemistry, CS)")
print(f"  Sessional marks   : 3 per subject (out of 30 each)")
print(f"  Final exam        : 1 per subject (out of 70)")
print(f"  Attendance        : {attendance.min():.1f}% – {attendance.max():.1f}%  (avg {attendance.mean():.1f}%)")
print(f"  Past failures     : 0 – {past_failures.max()}")
print(f"  Overall result    : {overall.min():.1f} – {overall.max():.1f}  (avg {overall.mean():.1f})")
print(f"  Pass rate (>=40)  : {round((overall >= 40).mean() * 100, 1)}%")
print(f"  Columns           : {len(df.columns)}")
print()
print("  How to use in app:")
print("  1. Select 'Upload My Own CSV'")
print("  2. Upload synthetic_student_data.csv")
print("  3. Set 4 subjects:")
print("     Math     -> S1=Math_Sessional1, S2=Math_Sessional2,")
print("                 S3=Math_Sessional3, Final=Math_Final_Exam")
print("     Physics  -> same pattern with Physics_ columns")
print("     Chemistry-> same pattern with Chem_ columns")
print("     CS       -> same pattern with CS_ columns")
print("  4. Attendance -> Attendance_%  (Percentage 0-100)")
print("  5. Past Failures -> Past_Failures (count column)")
print("  6. Final Result  -> Overall_Result  (max=100)")
print()
print("  File saved: synthetic_student_data.csv")
print("=" * 55)
