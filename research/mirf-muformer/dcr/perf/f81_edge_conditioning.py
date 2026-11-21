"""Local physical and mixed-coordinate optimization for one fixed edge."""
from decimal import Decimal

ZERO = Decimal(0)
ONE = Decimal(1)
C1 = Decimal("1e-4")
GATE = Decimal("1e-5")


def redraw(s):
    if s < Decimal("1e-45"):
        return s * (ONE - s/2 + s*s/6)
    return ONE - (-s).exp()


class EdgeObjective:
    """Work on high-precision A/B odds, holding all other edges fixed."""

    def __init__(self, site_terms, initial):
        self.ratios = tuple(row[0]/row[1] for row in site_terms)
        self.zero_feasible = all(a > 0 for a in self.ratios)
        p = redraw(initial)
        self.reference_mass = tuple((ONE-p)*a+p for a in self.ratios)
        self.calls = 0

    def evaluate(self, s):
        self.calls += 1
        p = redraw(s)
        keep = (-s).exp()
        f = g = h = ZERO
        for a, old in zip(self.ratios, self.reference_mass):
            mass = keep*a+p
            if mass <= 0:
                return None
            score = keep*(a-ONE)/mass
            f -= (mass/old).ln()
            g += score
            h += score*(score-ONE)
        return tuple(x/len(self.ratios) for x in (f, g, h))


def physical(value):
    return Decimal.from_float(float(value))


def projected(s, gradient):
    return abs(min(s, gradient))


def condition(s, g, h, method):
    """Propose a bounded descent step, with explicit activation at s=0."""
    direct_trust = Decimal(".25")
    if not s:
        # One-sided activation. An observed slope of zero leaves the edge zero.
        scale = max(abs(h), Decimal("1e-8"))
        return None, min(direct_trust, -g/scale)
    if method == "direct_diag":
        scale = max(abs(h), Decimal("1e-8"))
        return None, max(-direct_trust, min(direct_trust, -g/scale))

    transformed_g = s*g
    transformed_h = s*s*h + transformed_g
    if method == "hybrid_log_diag":
        scale = max(abs(transformed_h), abs(transformed_g)/4)
        change = -transformed_g/scale if scale else ZERO
    elif method == "hybrid_log":
        change = ONE if transformed_g < 0 else -ONE
    else:
        raise ValueError(method)
    return s.ln(), change


def solve_edge(rows, initial_length, method, *, maxeval=96, target=GATE):
    """Optimize one branch with Armijo acceptance measured in physical coordinates."""
    step_log = []
    point = physical(initial_length)
    objective = EdgeObjective(rows, point)
    value, g, h = objective.evaluate(point)
    status = "local_budget"
    tried_boundary = False
    while objective.calls < maxeval:
        if projected(point, g) <= target:
            status = "local_projected_gradient"
            break

        z, d = condition(point, g, h, method)
        candidate = None
        # A discrete zero move needs both feasible support and the zero KKT sign.
        if point > 0 and g > 0 and objective.zero_feasible and not tried_boundary:
            boundary = objective.evaluate(ZERO)
            tried_boundary = True
            q = -point*g
            accepted = boundary[1] >= 0 and boundary[0] <= value+C1*q
            step_log.append(event(objective, point, ZERO, q, value, boundary, accepted, "boundary"))
            if accepted:
                candidate = ZERO, boundary

        if candidate is None:
            for trial in range(24):
                if objective.calls >= maxeval:
                    break
                rate = ONE/Decimal(2**trial)
                proposed = max(ZERO, point+rate*d) if z is None else (z+rate*d).exp()
                s = physical(proposed)
                q = g*(s-point)
                if q >= 0 or (not s and not objective.zero_feasible):
                    continue
                # Positive log coordinates must not round silently into the boundary state.
                if z is not None and not s:
                    continue
                result = objective.evaluate(s)
                accepted = result is not None and result[0] <= value+C1*q
                step_log.append(event(objective, point, s, q, value, result, accepted,
                                      "activation" if not point else ("log" if z is not None else "physical")))
                if accepted:
                    candidate = s, result
                    break
        if candidate is None:
            status = "local_budget" if objective.calls >= maxeval else "no_feasible_armijo_update"
            break
        point, (value, g, h) = candidate

    if projected(point, g) <= target:
        status = "local_projected_gradient"
    return {"method": method, "status": status, "start_length": float(initial_length),
            "end_length": float(point), "relative_objective": str(value), "gradient_s": str(g),
            "diagonal_s": str(h), "gradient_log_s": str(point*g) if point else None,
            "diagonal_log_s": str(point*point*h+point*g) if point else None,
            "projected_s": str(projected(point, g)), "zero_feasible": objective.zero_feasible,
            "evaluations": objective.calls, "accepted_updates": sum(x["accepted"] for x in step_log),
            "trials": step_log}


def event(objective, point, trial, q, value, result, accepted, phase):
    return {"evaluation": objective.calls, "mode": phase, "length_before": float(point),
            "length_trial": float(trial), "projected_slope": str(q),
            "armijo_rhs": str(value+C1*q), "objective": str(result[0]) if result is not None else None,
            "gradient_at_trial": str(result[1]) if result is not None else None, "accepted": accepted}
