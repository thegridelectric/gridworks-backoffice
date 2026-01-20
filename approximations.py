import matplotlib.pyplot as plt

# Parameters

ALPHA=12
BETA=-0.22
GAMMA = 0.002
INTERMEDIATE_POWER = 1.5
INTERMEDIATE_RSWT=130
DD_POWER = 12
DD_RSWT=180
DD_DELTA_T = 20
NO_POWER_RSWT = -ALPHA/BETA
if ALPHA != DD_POWER:
    raise ValueError("ALPHA and DD_POWER must be equal")

x0 = NO_POWER_RSWT
xi = INTERMEDIATE_RSWT
xd = DD_RSWT
y0 = 0
yi = INTERMEDIATE_POWER
yd = DD_POWER

c = (xi*xd)/(xi-xd) * ((yd*x0)/(xd*(x0-xd)) - (yi*x0)/(xi*(x0-xi)))
b = (yi*x0)/(xi*(x0-xi)) - (x0+xi)/(x0*xi)*c
a = -b/x0 - c/x0/x0

# Functions

def required_heating_power(oat, ws):
    r = ALPHA + BETA*oat + GAMMA*ws*(65-oat)
    return r if r>0 else 0

def delivered_heating_power(swt):
    d = a*swt**2 + b*swt + c
    return d if d>0 else 0

def required_swt(hp):
    c2 = c - hp
    return (-b + (b**2-4*a*c2)**0.5)/(2*a)

def delta_T(swt):
    d = DD_DELTA_T/DD_POWER * delivered_heating_power(swt)
    d = 0 if swt<NO_POWER_RSWT else d
    return d if d>0 else 0

# Required heating power and RSWT a a function of OAT

oats = sorted(list(range(-10,70,5)), reverse=True)
wss = [0]*len(oats)
df_0_wind = {
    'oat': oats,
    'ws': wss,
    'kw': [required_heating_power(x,y) for x,y in zip(oats,wss)],
    'rswt': [required_swt(required_heating_power(x,y)) for x,y in zip(oats,wss)],
    'deltaT': [delta_T(required_swt(required_heating_power(x,y))) for x,y in zip(oats,wss)],
}

wss = [10]*len(oats)
df_10_wind = {
    'oat': oats,
    'ws': wss,
    'kw': [required_heating_power(x,y) for x,y in zip(oats,wss)],
    'rswt': [required_swt(required_heating_power(x,y)) for x,y in zip(oats,wss)],
    'deltaT': [delta_T(required_swt(required_heating_power(x,y))) for x,y in zip(oats,wss)],
}

fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
ax.plot(df_0_wind['oat'], df_0_wind['kw'], label='Required heating, wind speed = 0 mph', alpha=0.7, color='tab:blue')
ax.plot(df_10_wind['oat'], df_10_wind['kw'], label='Required heating, wind speed = 10 mph', alpha=0.7, color='tab:blue', linestyle='--')
ax.set_ylabel('Required heating power [kW]')
ax2.plot(df_0_wind['oat'], df_0_wind['rswt'], color='tab:orange', label='RSWT, wind speed = 0 mph', alpha=0.7)
ax2.plot(df_10_wind['oat'], df_10_wind['rswt'], color='tab:orange', label='RSWT, wind speed = 10 mph', alpha=0.7, linestyle='--')
ax2.set_ylabel('RSWT [F]')
ax.set_xlabel('Outside air temperature [F]')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='best')
ax.grid(alpha=0.5)
plt.title('Required heating power and RSWT as a function of OAT')
plt.show()




# # Delta T based on SWT

# swt = list(range(int(NO_POWER_RSWT)-10,190))
# plt.plot(swt, [delta_T(x) for x in swt])
# plt.scatter(DD_RSWT, DD_DELTA_T, label="Design day", color='tab:red')
# plt.scatter(NO_POWER_RSWT, 0, label="No power SWT", color='tab:green')
# plt.scatter(INTERMEDIATE_RSWT, delta_T(INTERMEDIATE_RSWT), label="Intermediate", color='tab:orange')
# plt.title("Temperature drop as a function of SWT")
# plt.xlabel("SWT [F]")
# plt.ylabel("Delta T [F]")
# plt.grid()
# plt.legend()
# plt.plot()

