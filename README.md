### Fluids Management Project

The purpose of fluids management is two fold--coordinating haul frequency and
spill prevention for both oil and gas. The output of the program is requesting
gauges (measurements) by humans or hauls (liquid takeaway by trucks).

The current algorithm has some large limitations. For liquid projection it uses
the last previous calculated liquid rate per day based on the last two gauges.
For spill prevention it applies this rate to the last most full tank as it assumes
worst case is all liquid goes into that tank.

The new algorithm will include:<br />
    •	Historical production rates by gauges (not just the last gauge to gauge)<br />
    •	Liquid gas ratio<br />
    •	Historical haul frequency<br />
    •	Any data that shows artificial lift was installed or optimized<br />

One requirement will be to map out wells to their associated tanks and weight how
liquids are being distributed across multiple tanks. Next would be to find a
specific algorithm to accurately predict when hauls are necessary based on
production rates. We will also look into whether increased gauge frequency is
necessary based on current production rates. Being able to link dates when the
artificial lift has changed will allow reasonable expectations for when production
rates will increase, which may result in the need for frequent gauging.

Another project would be identifying the ratio of liquid rates between shared
tanks-that is, when a well dumps into a series of tanks, what ratio of the
liquid goes into tank 1 and what goes into tank 2-if we knew this we could get
away from assuming that 100% goes to each tank and taking the worst case scenario.
