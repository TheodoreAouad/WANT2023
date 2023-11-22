from dataclasses import dataclass

import numpy as np
import cvxpy as cp


@dataclass
class IterativeProjectionBase:
    Wini: np.ndarray
    bini: float
    S: np.ndarray
    operation: str = "dilation"

    @property
    def N(self):
        return len(self.Wini)

    def solve(self, *args, **kwargs):
        self.Wvar = cp.Variable(self.Wini.shape)
        self.bvar = cp.Variable(1)

        if self.operation in ["dilation", "union"]:
            self.constraints = self.dilation_constraints(self.Wvar, self.bvar, self.S)
        elif self.operation in ["erosion", "intersection"]:
            self.constraints = self.erosion_constraints(self.Wvar, self.bvar, self.S)
        else:
            raise ValueError("operation must be dilation or erosion")

        self.objective = cp.Minimize(1/2 * cp.sum_squares(self.Wvar - self.Wini) + 1/2 * cp.sum_squares(self.bvar - self.bini))
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob.solve(*args, **kwargs)

        return self

    def dilation_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        return []

    def erosion_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        return []

    @property
    def value(self):
        return self.prob.value

    @property
    def W(self):
        return self.Wvar.value

    @property
    def b(self):
        return self.bvar.value

    @property
    def loss(self):
        return self.prob.value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"Loss={self.loss:.4e} "
            ")"
        )


class IterativeProjectionPositive(IterativeProjectionBase):
    def dilation_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        self.constraint0 = [cp.sum(Wvar[~S]) <= bvar]
        self.constraintsT = [bvar <= Wvar[S]]
        self.constraintsK = [Wvar >= 0]
        return self.constraint0 + self.constraintsT + self.constraintsK

    def erosion_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        self.constraint0 = [cp.sum(Wvar[S]) >= bvar]
        self.constraintsT = [cp.sum(Wvar) - Wvar[S] <= bvar]
        self.constraintsK = [Wvar >= 0]
        return self.constraint0 + self.constraintsT + self.constraintsK

    @property
    def K(self):
        return np.isclose(self.W, 0) & ~self.S

    @property
    def Kbar(self):
        return ~self.S & ~self.K

    @property
    def T(self):
        return np.isclose(self.W, self.b) & self.S

    @property
    def lambda0(self):
        return self.constraint0[0].dual_value

    @property
    def lambdat(self):
        return np.array([constraint.dual_value for constraint in self.constraintsT])

    @property
    def lambdak(self):
        return np.array([constraint.dual_value for constraint in self.constraintsK])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"Loss={self.loss:.4e} "
            f"Kbar: {self.Kbar.sum()} K: {self.K.sum()} T: {self.T.sum()}"
            ")"
        )


class AnalyticalProjectionOnSelem:

    def __init__(
        self,
        Wini: np.ndarray = None,
        bini: np.ndarray = None,
        S: np.ndarray = None,
        T: np.ndarray = None,
        K: np.ndarray = None,
        epsilon: float = 1e-4
    ):
        self.Wini = Wini
        self.bini = bini
        self.S = S
        self.T = T
        self.K = K
        self.epsilon = epsilon

        self.b = None
        self.lambda0: float = None
        self.lambdat: np.ndarray = None
        self.lambdak: np.ndarray = None
        self.W = None
        self.loss = None
        self.summary = None

    @staticmethod
    def left_cond(Wini: np.ndarray = None, bini: np.ndarray = None, S: np.ndarray = None, epsilon: float = 1e-4):
        lb = Wini[~S].sum()
        return lb  <= bini * (1 + epsilon)  # Small relaxation for numerical errors

    @staticmethod
    def right_cond(Wini: np.ndarray = None, bini: np.ndarray = None, S: np.ndarray = None, epsilon: float = 1e-4):
        ub = Wini[S].min()
        return ub >= bini * (1 - epsilon)  # Small relaxation for numerical errors

    @staticmethod
    def left_margin(Wini: np.ndarray = None, bini: np.ndarray = None, S: np.ndarray = None):
        lb = Wini[~S].sum()
        return bini - lb

    @staticmethod
    def right_margin(Wini: np.ndarray = None, bini: np.ndarray = None, S: np.ndarray = None):
        ub = Wini[S].min()
        return ub - bini

    @staticmethod
    def positive_margin(Wini: np.ndarray = None):
        return Wini.min()

    @staticmethod
    def positive_cond(Wini: np.ndarray = None, epsilon: float = 1e-4):
        return Wini.min() >= -epsilon

    def lagrangian_cond(self,):
        if self.lambdat is not None and (self.lambdat[self.T] < -self.epsilon).any():
            return False

        if self.lambda0 is not None and self.lambda0 < -self.epsilon:
            return False

        if self.lambdak is not None and (self.lambdak[self.K] < -self.epsilon).any():
            return False

        return True


    def get_summary(self, Wini, bini, S, T=None, K=None):
        return {
            "b": self.b,
            "lambda0": self.lambda0,
            "loss": self.loss,
            "lambdat": self.lambdat[T] if self.lambdat is not None else None,
            "lambdak": self.lambdak[K] if self.lambdak is not None else None,
            "W": self.W,
            "conds_init_left": self.left_cond(Wini=Wini, bini=bini, S=S),
            "conds_init_right": self.right_cond(Wini=Wini, bini=bini, S=S),
            "conds_init_positive": self.positive_cond(Wini=Wini),
            "margins_init_left": self.left_margin(Wini=Wini, bini=bini, S=S),
            "margins_init_right": self.right_margin(Wini=Wini, bini=bini, S=S),
            "margins_init_positive": self.positive_margin(Wini=Wini),
            "conds_proj_left": self.left_cond(Wini=self.W, bini=self.b, S=S),
            "conds_proj_right": self.right_cond(Wini=self.W, bini=self.b, S=S),
            "conds_proj_positive": self.positive_cond(Wini=self.W),
            "margins_proj_left": self.left_margin(Wini=self.W, bini=self.b, S=S),
            "margins_proj_right": self.right_margin(Wini=self.W, bini=self.b, S=S),
            "margins_proj_positive": self.positive_margin(Wini=self.W),
            "lagrangian_cond": self.lagrangian_cond(),
        }

    def projection_left_right(self,):
        Wini, bini, T, K, S = self.Wini, self.bini, self.T, self.K, self.S

        card_sbar = (~S).sum()
        card_T = T.sum()
        card_K = K.sum()
        card_Kbar = card_sbar - card_K
        wsum_T = Wini[T].sum()
        wsum_Kbar = Wini[~S & ~K].sum()

        denom = np.float32(1 / (card_Kbar * (card_T  + 1) + 1))

        self.b = denom * (wsum_Kbar + card_Kbar * (wsum_T + bini))

        self.lambdat = np.zeros_like(Wini)
        self.lambdat[T] = self.b - Wini[T]

        self.lambda0 = denom * ((card_T + 1) * wsum_Kbar - wsum_T - bini)

        self.lambdak = np.zeros_like(Wini)
        self.lambdak[K] = self.lambda0 - Wini[K]

        W = np.zeros_like(Wini)
        W[T] = self.b
        W[S & ~T] = Wini[S & ~T]
        W[~K & ~S] = Wini[~K & ~S] - self.lambda0
        self.W = W

        # self.loss = ((Wini[T] - self.b)**2).sum() + card_sbar * self.lambda0 ** 2 + (self.b - bini) ** 2
        self.loss = 1/2 * (((self.W - Wini) ** 2).sum() + (self.b - bini) ** 2)

        self.summary = self.get_summary(Wini=Wini, bini=bini, S=S, T=T)

        return self

    @property
    def Kbar(self):
        return ~self.S & ~self.K
