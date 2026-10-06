import AdaCrowd.Core
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic

/-! The probability-space version. Square integrability and integrability of
objective values make every expectation in the manuscript well-defined. -/

open MeasureTheory
open scoped BigOperators RealInnerProductSpace

namespace AdaCrowd
namespace General

variable {Ω E : Type*} [mΩ : MeasurableSpace Ω] {μ : Measure Ω}
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem integrable_sq (a : Lp E 2 μ) : Integrable (fun ω => ‖a ω‖ ^ 2) μ := by
  simpa only [real_inner_self_eq_norm_sq] using L2.integrable_inner (𝕜 := ℝ) a a

variable [CompleteSpace E]

/-- A2 in its ordinary conditional-expectation form implies cancellation
against any square-integrable vector measurable with respect to the past. -/
theorem conditional_inner_zero [IsFiniteMeasure μ]
    {m : MeasurableSpace Ω} (hm : m ≤ mΩ) (a z : Lp E 2 μ)
    (ha : AEStronglyMeasurable[m] a μ)
    (hz : μ[(fun ω => z ω) | m] =ᵐ[μ] 0) :
    (∫ ω, ⟪a ω, z ω⟫ ∂μ) = 0 := by
  letI : MeasurableSpace Ω := mΩ
  have he := (Lp.memLp z).condExpL2_ae_eq_condExp (𝕜 := ℝ) hm
  rw [Lp.toLp_coeFn] at he
  have hz' : (condExpL2 E ℝ hm z : Lp E 2 μ) = 0 := by
    apply Lp.ext
    exact (he.trans hz).trans (Lp.coeFn_zero E 2 μ).symm
  have hi := inner_condExpL2_eq_inner_fun (𝕜 := ℝ) hm z a ha
  rw [hz', inner_zero_left] at hi
  rw [← L2.inner_def, real_inner_comm]
  exact hi.symm

/-- Integrate the smoothness inequality and derive one-step expected descent. -/
theorem expected_descent [IsFiniteMeasure μ]
    {m : MeasurableSpace Ω} (hm : m ≤ mΩ)
    (a b z : Lp E 2 μ) (ha : AEStronglyMeasurable[m] a μ)
    (hb : AEStronglyMeasurable[m] b μ)
    (hz : μ[(fun ω => z ω) | m] =ᵐ[μ] 0)
    {v vnext epsilon : Ω → ℝ} (hv : Integrable v μ) (hvn : Integrable vnext μ)
    (heps : Integrable (fun ω => epsilon ω ^ 2) μ)
    {L h sigma2 : ℝ} (hL : 0 ≤ L) (hh : 0 ≤ h) (hstep : L * h ≤ 1)
    (hvariance : (∫ ω, ‖z ω‖ ^ 2 ∂μ) ≤ sigma2)
    (hbias : ∀ᵐ ω ∂μ, ‖b ω‖ ≤ epsilon ω)
    (hsmooth : ∀ᵐ ω ∂μ, vnext ω ≤ v ω
      - h * ⟪a ω, a ω + b ω + z ω⟫
      + L * h ^ 2 / 2 * ‖a ω + b ω + z ω‖ ^ 2) :
    (∫ ω, vnext ω ∂μ) ≤ (∫ ω, v ω ∂μ)
      - h / 2 * (∫ ω, ‖a ω‖ ^ 2 ∂μ)
      + h / 2 * (∫ ω, epsilon ω ^ 2 ∂μ) + L * h ^ 2 / 2 * sigma2 := by
  letI : MeasurableSpace Ω := mΩ
  have hpoint : ∀ᵐ ω ∂μ, vnext ω ≤ v ω - h / 2 * ‖a ω‖ ^ 2
      + h / 2 * ‖b ω‖ ^ 2 + L * h ^ 2 / 2 * ‖z ω‖ ^ 2
      - h * ⟪a ω, z ω⟫ + L * h ^ 2 * (⟪a ω, z ω⟫ + ⟪b ω, z ω⟫) := by
    filter_upwards [hsmooth] with ω hω
    have hd := noisy_direction_bound (a ω) (b ω) (z ω) hh hstep
    rw [inner_add_left] at hd
    linarith
  have ha2 := integrable_sq a
  have hb2 := integrable_sq b
  have hz2 := integrable_sq z
  have haz := L2.integrable_inner (𝕜 := ℝ) a z
  have hbz := L2.integrable_inner (𝕜 := ℝ) b z
  have h1 := hv.sub (ha2.const_mul (h / 2))
  have h2 := h1.add (hb2.const_mul (h / 2))
  have h3 := h2.add (hz2.const_mul (L * h ^ 2 / 2))
  have h4 := h3.sub (haz.const_mul h)
  have h5 := (haz.add hbz).const_mul (L * h ^ 2)
  have hi := integral_mono_ae hvn (h4.add h5) hpoint
  simp only [Pi.add_apply, Pi.sub_apply] at hi
  have e1 := integral_sub hv (ha2.const_mul (h / 2))
  have e2 := integral_add h1 (hb2.const_mul (h / 2))
  have e3 := integral_add h2 (hz2.const_mul (L * h ^ 2 / 2))
  have e4 := integral_sub h3 (haz.const_mul h)
  have e5 := integral_add h4 h5
  have e6 := integral_add haz hbz
  simp only [Pi.add_apply, Pi.sub_apply] at e1 e2 e3 e4 e5 e6
  rw [e5, e4, e3, e2, e1] at hi
  simp only [integral_const_mul, e6] at hi
  have hza := conditional_inner_zero hm a z ha hz
  have hzb := conditional_inner_zero hm b z hb hz
  rw [hza, hzb] at hi
  have hib := integral_mono_ae hb2 heps (by
    filter_upwards [hbias] with ω hω
    nlinarith [norm_nonneg (b ω)])
  have hib' := mul_le_mul_of_nonneg_left hib (show 0 ≤ h / 2 by positivity)
  have hiv' := mul_le_mul_of_nonneg_left hvariance (show 0 ≤ L * h ^ 2 / 2 by positivity)
  linarith

end General
end AdaCrowd
