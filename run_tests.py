"""Small deterministic checks for the submission reproducibility package."""
import sys
sys.path.insert(0, "src")
import numpy as np
from rcda_moea.algorithm import RCDA_MOEA, RCDAConfig
from baselines.nsga2 import NSGA2, NSGA2Config
from rcda_moea.selection import hypervolume_2d
from rcda_moea.diversity import compute_mutation_weights

rng = np.random.default_rng(0)
X = rng.normal(size=(30, 20))
ref = (2.0, 1.0)

rc = RCDA_MOEA(X, RCDAConfig(population_size=10, generations=3, ref_point=ref), 1)
A, L = rc.run(record_diagnostics=True)
assert len(A) > 0 and len(L) == 3 and len(rc.diagnostics) == 3
final = hypervolume_2d(np.array([x.objectives for x in A]), np.asarray(ref))
assert np.isclose(final, L[-1], rtol=1e-12, atol=1e-15)
assert all("hv_feedback" in d and "hv_post" in d for d in rc.diagnostics)
print("PASS 1/4 RCDA final archive HV == hv_log[-1]")

ns = NSGA2(X, NSGA2Config(population_size=10, generations=3, ref_point=ref), 1)
B, M = ns.run()
final_ns = hypervolume_2d(np.array([x.objectives for x in B]), np.asarray(ref))
assert np.isclose(final_ns, M[-1], rtol=1e-12, atol=1e-15)
print("PASS 2/4 NSGA-II final archive HV == hv_log[-1]")

wr, wc = compute_mutation_weights(0.2, 0.8, 1e-3, sharpness=0.0)
assert np.isclose(wr, 0.5) and np.isclose(wc, 0.5)
print("PASS 3/4 disabled structural feedback gives W_R=W_C=0.5")

# Same algorithm seed/config must be deterministic.
A2, L2 = RCDA_MOEA(X, RCDAConfig(population_size=10, generations=3, ref_point=ref), 1).run()
assert np.allclose(L, L2)
print("PASS 4/4 deterministic rerun")
