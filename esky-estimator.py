import stan
import numpy as np

code = """
data {
    int<lower=0> N;  // Number of trials
    vector[N] can_diameters;  // Diameters of cans in cm
    vector[N] can_heights;  // Heights of cans in cm
    real esky_length;  // Length of esky in cm
    real esky_width;  // Width of esky in cm
    real esky_height;  // Height of esky in cm
}

parameters {
    real<lower=0, upper=1> pack_probability;  // Probability of being able to pack a can
}

model {
    pack_probability ~ beta(2, 2);  // Prior distribution
    for (n in 1:N) {
        real volume_can = 3.14 * pow(can_diameters[n] / 2, 2) * can_heights[n];
        real volume_esky = esky_length * esky_width * esky_height;
        // Likelihood
        if (volume_can <= volume_esky) {
            1 ~ bernoulli(pack_probability);
        } else {
            0 ~ bernoulli(pack_probability);
        }
    }
}
"""

# Generate some synthetic data
N = 100
can_diameters = np.random.uniform(6, 10, N)
can_heights = np.random.uniform(10, 20, N)
esky_length = 40
esky_width = 30
esky_height = 25

data = {
    "N": N,
    "can_diameters": can_diameters,
    "can_heights": can_heights,
    "esky_length": esky_length,
    "esky_width": esky_width,
    "esky_height": esky_height,
}

post = stan.build(code, data=data)
fit = post.sample(num_chains=4, num_samples=1000)

results = fit["pack_probability"]
pack_probability = np.mean(results)

print(f"Estimated Probability of Packing a Can: {pack_probability}")
