from pydantic import BaseModel

class HouseParameters(BaseModel):
    alpha: float
    beta: float
    gamma: float
    intermediate_power_kw: float
    intermediate_rswt: float
    dd_power_kw: float
    dd_rswt: float
    dd_delta_t: float
    
    def to_dict(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "intermediate_power_kw": self.intermediate_power_kw,
            "intermediate_rswt": self.intermediate_rswt,
            "dd_power_kw": self.dd_power_kw,
            "dd_rswt": self.dd_rswt,
            "dd_delta_t": self.dd_delta_t,    
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)