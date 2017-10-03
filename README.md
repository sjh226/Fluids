### Fluids Management Project

The purpose of fluids management is two fold--coordinating haul frequency and
spill prevention for both oil and gas. The output of the program is requesting
gauges (measurements) by humans or hauls (liquid takeaway by trucks).

The current algorithm:

For liquid projection it uses the last previous calculated liquid rate per day
based on the last two gauges.

For spill prevention it applies this rate to the
last most full tank as it assumes worst case is all liquid goes into that tank.

The new algorithm will include:
    •	Historical production rates by gauges (not just the last gauge to gauge)
    •	Liquid gas ratio (Pritens project)
    •	Historical haul frequency (Arkoma’s project)
    •	Any data that shows artificial lift was installed or optimized

Another project would be identifying the ratio of liquid rates between shared
tanks-that is, when a well dumps into a series of tanks, what ratio of the
liquid goes into tank 1 and what goes into tank 2-if we knew this we could get
away from assuming that 100% goes to each tank and taking the worst case scenario.
