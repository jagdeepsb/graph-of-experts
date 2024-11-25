import random
import numpy as np

class PiecewiseFunction:
    def __init__(self, lower_bound=-1.0, upper_bound=1.0, num_splits=4):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_splits = num_splits
        
        # Compute interval boundaries based on the number of splits
        self.intervals = np.linspace(lower_bound, upper_bound, num_splits + 1)
        
        # Randomly initialize slope and intercept for each interval
        self.slopes = [random.uniform(-1, 1) for _ in range(num_splits)]
        self.intercepts = [random.uniform(-1, 1) for _ in range(num_splits)]
        
    def __call__(self, x):
        # Ensure x is within the specified range
        if not (self.lower_bound <= x <= self.upper_bound):
            raise ValueError(f"Input x must be within the range [{self.lower_bound}, {self.upper_bound}]")
        
        # Determine which interval x belongs to and apply the corresponding linear function
        for i in range(self.num_splits):
            if self.intervals[i] <= x < self.intervals[i + 1]:
                return self.slopes[i] * x + self.intercepts[i]
        
        # Handle the edge case where x == upper_bound
        return self.slopes[-1] * x + self.intercepts[-1]
    
class PolynomialFunction:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        
    def __call__(self, x):
        return sum([coeff * x ** i for i, coeff in enumerate(self.coefficients)])
    
class SinusoidalFunction:
    def __init__(self, amplitude, frequency, phase):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        
    def __call__(self, x):
        return self.amplitude * np.sin(self.frequency * x + self.phase)

if __name__ == "__main__":
    piecewise_func = PiecewiseFunction(lower_bound=-1, upper_bound=1, num_splits=4)
    print(piecewise_func(0.5))
    print(piecewise_func(-0.5))
    print(piecewise_func(1.0))
    print(piecewise_func(-1.0))