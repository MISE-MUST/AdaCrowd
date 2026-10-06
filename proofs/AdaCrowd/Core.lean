import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity

/-! The analytic and telescoping parts of the TPAMI stationarity proof. -/

open scoped BigOperators RealInnerProductSpace

namespace AdaCrowd

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A1 is the quadratic upper-model inequality used in the manuscript. -/
def SmoothUpperModel (F : E → ℝ) (grad : E → E) (L : ℝ) : Prop :=
  ∀ x y, F y ≤ F x + ⟪grad x, y - x⟫ + L / 2 * ‖y - x‖ ^ 2

theorem smooth_update {F : E → ℝ} {grad : E → E} {L h : ℝ}
    (hsmooth : SmoothUpperModel F grad L) (hh : 0 ≤ h) (x g : E) :
    F (x - h • g) ≤ F x - h * ⟪grad x, g⟫ + L * h ^ 2 / 2 * ‖g‖ ^ 2 := by
  have hs := hsmooth x (x - h • g)
  have he : x - h • g - x = -(h • g) := by abel
  rw [he, inner_neg_right, real_inner_smul_right, norm_neg, norm_smul,
    Real.norm_eq_abs, abs_of_nonneg hh] at hs
  nlinarith [hs]

/-- Completion of the square gives the sharper coefficient 1 on subset bias. -/
theorem biased_direction_bound (a b : E) {L h : ℝ}
    (hh : 0 ≤ h) (hstep : L * h ≤ 1) :
    -h * ⟪a, a + b⟫ + L * h ^ 2 / 2 * ‖a + b‖ ^ 2
      ≤ -h / 2 * ‖a‖ ^ 2 + h / 2 * ‖b‖ ^ 2 := by
  have hn := norm_add_sq_real a b
  have hp := mul_nonneg (mul_nonneg hh (sub_nonneg.mpr hstep)) (sq_nonneg ‖a + b‖)
  rw [inner_add_right, real_inner_self_eq_norm_sq]
  nlinarith

/-- One stochastic step, with the two zero-mean cross terms still explicit. -/
theorem noisy_direction_bound (a b z : E) {L h : ℝ}
    (hh : 0 ≤ h) (hstep : L * h ≤ 1) :
    -h * ⟪a, a + b + z⟫ + L * h ^ 2 / 2 * ‖a + b + z‖ ^ 2
      ≤ -h / 2 * ‖a‖ ^ 2 + h / 2 * ‖b‖ ^ 2
        + L * h ^ 2 / 2 * ‖z‖ ^ 2
        - h * ⟪a, z⟫ + L * h ^ 2 * ⟪a + b, z⟫ := by
  have hb := biased_direction_bound a b hh hstep
  rw [inner_add_right, norm_add_sq_real]
  linarith

/-- Sum a proved expected-descent recurrence; V, G, B denote expectations. -/
theorem constant_step_stationarity {T : ℕ} (hT : 0 < T)
    {V G B : ℕ → ℝ} {h L sigma2 lower : ℝ} (hh : 0 < h)
    (hdescent : ∀ t < T, V (t + 1) ≤
      V t - h / 2 * G t + h / 2 * B t + L * h ^ 2 / 2 * sigma2)
    (hlower : lower ≤ V T) :
    (∑ t ∈ Finset.range T, G t) / T ≤
      2 * (V 0 - lower) / (h * T) + L * h * sigma2
        + (∑ t ∈ Finset.range T, B t) / T := by
  have hs := Finset.sum_le_sum (s := Finset.range T) (fun t ht =>
    show h / 2 * G t ≤ V t - V (t + 1) + h / 2 * B t + L * h ^ 2 / 2 * sigma2 by
      linarith [hdescent t (Finset.mem_range.mp ht)])
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum,
    Finset.sum_range_sub', Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hs
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  apply (div_le_iff₀ hTr).2
  apply (mul_le_mul_left (show 0 < h / 2 by positivity)).mp
  calc
    h / 2 * ∑ t ∈ Finset.range T, G t
        ≤ V 0 - lower + h / 2 * (∑ t ∈ Finset.range T, B t)
          + T * (L * h ^ 2 / 2 * sigma2) := by linarith
    _ = h / 2 * ((2 * (V 0 - lower) / (h * T) + L * h * sigma2
          + (∑ t ∈ Finset.range T, B t) / T) * T) := by
      field_simp
      ring

/-- Substitute h = eta / sqrt(T) into the constant-step theorem. -/
theorem sqrt_step_stationarity {T : ℕ} (hT : 0 < T)
    {V G B : ℕ → ℝ} {eta L sigma2 lower : ℝ} (heta : 0 < eta)
    (hdescent : ∀ t < T, V (t + 1) ≤
      V t - (eta / Real.sqrt T) / 2 * G t
        + (eta / Real.sqrt T) / 2 * B t
        + L * (eta / Real.sqrt T) ^ 2 / 2 * sigma2)
    (hlower : lower ≤ V T) :
    (∑ t ∈ Finset.range T, G t) / T ≤
      2 * (V 0 - lower) / (eta * Real.sqrt T)
        + L * eta * sigma2 / Real.sqrt T
        + (∑ t ∈ Finset.range T, B t) / T := by
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  have hsqrt : 0 < Real.sqrt (T : ℝ) := Real.sqrt_pos.2 hTr
  have hsquare := Real.sq_sqrt (le_of_lt hTr)
  have hb := constant_step_stationarity hT (div_pos heta hsqrt) hdescent hlower
  have hdenom : eta / Real.sqrt T * T = eta * Real.sqrt T := by
    field_simp
    nlinarith
  rw [hdenom] at hb
  convert hb using 1
  ring

theorem sqrt_step_admissible {T : ℕ} (hT : 0 < T) {L eta : ℝ}
    (hL : 0 < L) (heta : 0 < eta) (hetamax : eta ≤ 1 / L) :
    0 < eta / Real.sqrt T ∧ L * (eta / Real.sqrt T) ≤ 1 := by
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  have hs : 0 < Real.sqrt (T : ℝ) := Real.sqrt_pos.2 hTr
  have hTone : (1 : ℝ) ≤ T := by exact_mod_cast hT
  have hsone : (1 : ℝ) ≤ Real.sqrt T := by
    simpa using Real.sqrt_le_sqrt hTone
  have heL : eta * L ≤ 1 := (le_div_iff₀ hL).mp hetamax
  refine ⟨div_pos heta hs, ?_⟩
  rw [← mul_div_assoc]
  apply (div_le_iff₀ hs).2
  nlinarith

end AdaCrowd
