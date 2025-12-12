## V2.4 as of 23 July 2025

Vocab size: 82
Epochs: 5
Tokeniser: Char

Test 1

- "Run the settlement PDF for: F3-R09q-SM-30, F3-R26y-SM-55, F3-R55c-SM-74, F3-R81v-SM-26, F3-R82t-SM-20. SCD 2025-09-07, ASD 2025-09-10, max date 2025-11-17."
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R01p-SM-30', 'F3-R20b-SM-55', 'F3-R51f-SM-5', 'F3-R97i-SM-75', 'F3-R97y-SM-20'], 'SCD': '2025-09-01', 'ASD': '2025-09-01', 'max_date': '2025-09-09'}}

Test 2

- "What's the predicted settlement for F3-R10b-SM-11? Surcharge was Feb 10 2025, assess Feb 24 to Apr 15"
- {'function': 'Asaoka_data', 'args': {'ids': 'F3-R10b-SM-59', 'SCD': '2025-10-01', 'ASD': '2025-10-01', 'max_date': '2025-01-04'}}

Test 3

- "I need a document buddy for F3-R10z-SM-39, from start date 22nd July 2025, etc"
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R33f-SM-39'], 'SCD': '2025-12-29', 'ASD': '2025-02-1', 'max_date': '2025-03-1'}}

Overall Validation:
100 values tested
100% syntax
67% correct function identified
10% correct parameters identified

Remarks:

1. Complete Vocab size to 45x4. data set size would need to go 10k -> 100k.

2. Turn model into a classifier. classifies prompt into a specific model then extracts and or asks user for parameter values.

ex. What's the predicted settlement for F3-R10b-SM-11? Surcharge was Feb 10 2025, assess Feb 24 to Apr 15

classified to Asoka_data
[THEN] extract args from prompt using regex or similar.
[OR] extract args by asking user if needed.

## V3.0 as of 25 July 2025

Test 1

- "Hi system, initiate doc generation for F3-R22c-SM-77, F3-R37d-SM-23, F3-R36a-SM-22, F3-R10d-SM-35, F3-R38a-SM-41. Assessment phase Oct 19, 2024 - Aug 16, post surcharge 14 October. swjgfloqairgqhulerwough;oarghflaoirhoauherwgilaglaergu"
- reporter_Asaoka

Test 2

- "give me the detailed asaoka pdf for the plate F3-R10a-SM-01"                                             
- plot_combi_S

Test 3

- "Show me the current Asaoka assessment for plate F3-R06a-SM-30."               
- Asaoka_data

Overall Validation:
100 values tested
74% accuracy
98% date validation accuracy

Remarks: 

1. Larger dataset with more accurate prompts.

2. Connect to a larger model for validation.

## LLM update as of 25th July 2025

# prompt:
<<SYSTEM>>
You are MathGeniusGPT, an ultra‑reliable, friendly calculus tutor. Always:
• Stay 100% factual—no made‑up history or bogus formulas.  
• Define every symbol you use.  
• When you state a theorem, give the exact hypotheses (e.g. “0/0 or ∞/∞ indeterminate forms”) and the precise conclusion.  
• Provide one short example at the end.

<<USER>>
Hey there! I need a clear, correct explanation of L’Hôpital’s Rule in calculus. Please:

1. State the theorem exactly, including when it applies (indeterminate forms).  
2. Give the proof sketch in plain English (no hand‑wavy junk).  
3. Show the step‑by‑step limit computation for  
   \[
     \lim_{x\to 0} \frac{\sin(x)}{x}
   \]
   using L’Hôpital’s Rule.  
4. Provide a one‑sentence “real‑world” analogy so I remember it.  

Be concise but complete—no extra lore about “Lord de Lor” or random series. Let’s go!  

<<ASSISTANT>>

# Llama 2 : 7B : Q4

- Answer:
Hello there! *grins* Of course, I'd be happy to help you understand L'Hopital's Rule in calculus. Here we go! ð

1. theorem: L'Hopital's Rule states that if $\lim_{x\to a} f(x)$ exists, then $\lim_{x\to a} \frac{f(x)}{g(x)} = \lim_{x\to a} \frac{f'(x)}{g'(x)}$ for any continuous function $g(x)$ such that $g(a) = 0$.

2. proof sketch: We can use the definition of a limit to show that $\lim_{x\to a} f(x)/g(x) = \lim_{x\to a} (f(x) - g(x)) / g(x)$. Since $g(x)$ is continuous and $g(a) = 0$, we know that the limit of $(f(x) - g(x))/g(x)$ exists, say $\lim_{x\to a} (f(x) - g(x)) / g(x) = L$. Then, using the definition of a limit again, we have $\lim_{x\to a} f(x)/g(x) = \lim_{x\to a} (f(x) - g(x)) / g(x) = L$.

3. step-by-step limit computation for $\lim_{x\to 0} \frac{\sin(x)}{x}$ using L'Hopital's Rule:

We want to find the limit of $\sin(x)/x$ as $x\to 0$. We can use L'Hopital's Rule with $f(x) = \sin(x)$ and $g(x) = x$. Then, $f'(x) = \cos(x)$, and $g'(x) = 1$. So, using L'Hopital's Rule, we have:

$$\lim_{x\to 0} \frac{\sin(x)}{x} = \lim_{x\to 0} \frac{\sin(x)}{x} / \frac{1}{x} = \lim_{x\to 0} \frac{\cos(x)}{1} = \infty$$

4. real-world analogy: Imagine you're on a never-ending staircase, going up one step at a time. As you climb, the air around you starts to feel denser and denser, like a thick fog. L'Hopital's Rule is like a flashlight that helps you see through the fog and determine how fast you're actually moving. In this case, the faster you move, the more dense the fog becomes, until you finally reach the top step!

There, wasn't that fun and easy to understand? ð Please let me know if you have any questions or need further clarification!

- Time taken: 180.23 s

Remarks: 
- Hallucinatory but detailed
- Slow (2-3min for full generation)

# TinyLlama

- Answer
Sure! Here's a clear, correct explanation of LâHÃ´pital's Rule in calculus and a stepâbyâstep limit computation for the expression:  
1. L'HÃ´pital's Rule states that if a function is differentiable at a point and there exists a nonzero constant k such that 0 < k <= 1 and âf(y)/âx < k, then f'(x) <= Î± = k|â/âx| for all x â [Î± |âf(y)|, â« (Î± |âf(y)|^n)|d|y]. 2. Here, y has been replaced with any differentiable function of x using the integral by parts formula (i.e., the fact that the integral of a derivative over an area element I is equal to its value at the point where the area element is plotted). 3. The step by step limit computation for f'(x) evaluated at a given x and in the given range [Î± |âf(y)|, â« (Î± |âf(y)|^n)|d|y] uses L'HÃ´pital's Rule to write out all possible combinations of y values that result in a derivative at x greater or equal than Î± and less than or equal to the value of f evaluated at x. 4. This stepwise simplification is a useful analogue for some real-world situations, where we can use L'HÃ´pital's Rule to evaluate derivatives of nonlinear functions or compute limit inequalities for complicated series.

- Time taken:  26.22 s

Remarks:
- Def NOT using this
- Broken UTF
- basically useless at reasoning

# phi 4 : reasoning : 3.8b

- Answer:
<think>
Okay, let's tackle this request step by step. The user wants an explanation of L'HÃ´pital's Rule from MathGeniusGPT. Let's start by recalling what L'HÃ´pital's Rule is about.

First, I need to state the theorem exactly, including the conditions when it applies. From memory, L'HÃ´pital's Rule applies to limits that result in an indeterminate form, specifically 0/0 or â/â. The rule states that if such a limit of the ratio of two functions f(x)/g(x) approaches one of these forms as x approaches some point a, then the limit can be replaced by the limit of the ratio of their derivatives, provided that the latter limit exists. So, limââa [f(x)/g(x)] = limââa [f'(x)/g'(x)], given that this second limit exists.

Next, the user asked for a proof sketch in plain English. The standard proof involves using the Cauchy Mean Value Theorem. Here's how I remember it: If you have two functions f and g approaching limits L and M as x approaches a, then under certain conditions (like both tending to 0 or Â±â), the limit of their derivatives' ratio can be related back. But since they want this in plain English without hand-wavying, maybe explain that given limââa [f(x)/g(x)] = L/M (an indeterminate form), then after applying the quotient rule and rearranging terms using known limits, assuming certain conditions on f and g hold, we can apply L'HÃ´pital's Rule to get the derivatives' ratio.

Then, the specific example: compute limââ0 sin(x)/x using L'HÃ´pital's Rule. Wait, but when x approaches 0, both numerator and denominator approach 0, so it's a 0/0 indeterminate form. Applying L'HÃ´pital's Rule would mean taking the derivative of the numerator and denominator. The derivative of sin(x) is cos(x), and the derivative of x is 1. So the new limit becomes limââ0 cos(x)/1, which is cos(0) = 1. Hence, the answer should be 1.

But hold on, isn't there a known standard result that this limit is Ï/2? Wait no, wait: sin(x)/x as x approaches 0 is indeed 1. The derivative approach here works because applying L'HÃ´pital's directly gives cos(x)/1 which tends to 1. So that's correct.

Then the real-world analogy. Maybe something like if you have two rates approaching infinity or zero, their ratio can be replaced by looking at the rates of change? Or perhaps an analogy with velocity and time when acceleration is considered... Hmm. How else do we explain this? Maybe think of it as simplifying a problematic ratio (like 0/0) by converting it into something more manageable that's easier to evaluate directly, like taking the rate at which things are approaching those zeros.

Wait, another thought: maybe compare it to troubleshooting why two machines producing items both start slowing downâthe number of defects per hour goes to zero/zero. By looking at how fast they're slowing down (their rates), you can predict future performance by considering their derivatives. Not sure if that's perfect. Alternatively, in physics, when dealing with velocities approaching rest, the ratio of distances covered could be indeterminate, so taking accelerations into account simplifies it.

But I need a concise analogy. Let me think again: LâHÃ´pitalâs Rule is like having two runners slowing down (speed approaches zero). If you look at their distance apart over time instead, it's unclear why they're getting closer or fartherâit's an indeterminate form. But by looking at how fast they're decelerating (their derivatives), which might tell you that one is stopping while the other keeps moving slowly, you can figure out whether they'll catch up or not.

Alternatively, maybe a financial analogy: if two investments are both losing money to zero (rate approaching zero). The ratio of their values over time could be indeterminate. By looking at the rate at which they're losing value (derivatives), you get a clearer picture.

Hmm. Maybe better yet: Imagine you're trying to find out how quickly two shadows move towards each other as sunset, but both are moving in such a way that their speeds seem... frozen? Wait, no.

Alternatively, think of it as when icebergs meltâif the mass and height of an iceberg both decrease over time (approaching zero), you can't tell from just looking at them how fast they're melting. But by examining the rate at which each is decreasing (derivatives), you can understand their actual behavior better. Maybe that's okay.

Wait, maybe a more mathematical real-world example: suppose two cars are driving towards a point where they both stop eventually. The ratio of their distances from the stopping point to time might be an indeterminate form 0/0 as time approaches when they'll stop. By looking at how fast each car is decelerating (derivatives), you can determine that if one has higher acceleration, it would have stopped sooner... Hmm.

Alternatively, think of two people walking towards a meeting point: both walk infinitely slow down near the end so they meet on time. If you're measuring their progress as distance over remaining timeâboth approach zero/zero? Wait, maybe not.

Wait, perhaps an analogy from computing limits where variables are approaching extremes rather than zeros or infinities. For example, in physics, if two stars spiral towards each other in a galaxy; the ratio of their orbital radii to some time parameter might be indeterminate as they merge... Not sure.

Alternatively, think of baking cookies: you have enough dough (sin(x)) and cookie cutters sized x units wide. As x gets really small (approaching zero), how many cookies can you cut out? It's like â/â if the number of cutter sizes goes to infinity but each is tinyâLâHÃ´pitalâs would relate their rates.

Wait, maybe that's not ideal. Let's aim for something more relatable. Let me check: standard examples include sin(x)/x near zero being 1. The analogy could be approximating a small angle as sine â tangent â the angle itself (in radians), but that's trigonometry, which is why this limit gives 1.

Alternatively, use the idea that in engineering when designing materials under stress: if both force and displacement go to zero asymptotically. By looking at how quickly each changes with respect to some variable (dF/dD?), you can characterize their relationship rather than just the raw values.

I think I'll settle on an analogy where two quantities are approaching a problematic ratio, but by examining their rates of change, it's resolvedâlike tracking why two populations dying off: if both population and food supply go to zero as time passes, LâHÃ´pitalâs tells you look at how fast each is decreasing. If one decreases faster than the other per capita (derivatives), that affects survival.

But maybe a simpler analogy for memory's sake: Imagine trying to compare recipes where two ingredients are running outâthe same speedâit's like comparing their contributions but since they're both vanishing equally, by looking at how quickly theyâre being used up instead of directly the quantities left over. If oneâs supply is depleting faster because it's consumed quicker per recipe (derivative), that tells you which is more critical.

Alternatively, use a financial context: if two investments are depreciating in value to zero at different rates; evaluating their book value ratio might be indeterminate 0/0 as time approaches the end of depreciation. By looking at how fast each investment loses value (derivatives), one can assess whether theyâre losing more rapidly per unit over time.

In summary, after working through possible analogies and recalling standard teaching methods, perhaps best to use an analogy involving rates of change in real-world scenarios where direct evaluation isn't feasible but derivative analysis provides clarity. The key is that LâHÃ´pitalâs resolves 0/0 or â/â by switching the comparison from quantities themselves (which are problematic) to their changing rates.
</think>

**1. Statement of L'HÃ´pital's Rule:**  
If \(\lim_{x \to a} \frac{f(x)}{g(x)}\) is an indeterminate form \(0/0\) or \(\infty/\infty\), and both \(f(x)\) and \(g(x)\) are differentiable near \(a\) (except possibly at \(a\)), then:  
\[
\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)},
\]  
provided the latter limit exists.  

---

**2. Proof Sketch (Plain English):**  
L'HÃ´pital's Rule leverages derivative relationships to resolve \(0/0\) or \(\infty/\infty\) indeterminate forms. Assume:  
- If \(f(x)/g(x) \to 0/0\), then under conditions like "CÃ©saireâs Hypothesis" (both approaching 0 from the same side and their quotient ratio also approaches a finite limit \(L\)), we derive that \(\lim_{x \to a} \frac{f'(x)}{g'(x)} = L\).  
- If \(f(x)/g(x) \to \infty/\infty\), CÃ©saireâs Mean Value Theivative theorem applies, ensuring the limit of their derivative ratio exists.  

---

**3. Step-by-Step Computation for \(\lim_{x \to 0} \frac{\sin x}{x}\):**  
- **Step 1:** Confirm the form at \(x = 0\). Both \(\sin(0) = 0\) and denominator \(0\) yield \(0/0\), an indeterminate form.  
- **Step 2:** Apply L'HÃ´pital's Rule: take derivatives of numerator (\(f(x) = \sin x \implies f'(x) = \cos x\)) and denominator (\(g(x) = x \implies g'(x) = 1\)).  
- **Step 3:** Compute the new limit:  
  \[
  \lim_{x \to 0} \frac{\cos x}{1}.
  \]  
- **Step 4:** Evaluate directly. Since \(\cos(0) = 1\), the limit is \(1\).  

---

**4. Real-World Analogy:**  
Imagine two candles burning down: their heights (\(\sin x\)) and burn rates (\(x\)) both approach zero over time. LâHÃ´pitalâs Rule lets you compare how fast they're *shrinking* (derivatives) to determine if theyâll meet or drift apart, instead of just comparing current lengths.

**Final Answer:**  
The limit \(\lim_{x \to 0} \frac{\sin x}{x}\) is \(\boxed{1}\).

- Time taken:  559.28 s

Remarks:
- Math and reasoning is very accurate
- Kinda slow. (esp a lot if chopping off thinking)

# Deepseek : r1 : 1.5b

- Answer
<think>
Okay, so I need to explain LâHÃ´pitalâs Rule in calculus clearly and concisely. Let's start by stating the theorem exactly. It should include when it applies, specifically mentioning '0/0' or 'â/â' forms.

Next, I should provide a proof sketch without getting too technical. Maybe I'll think about how differentiation can help resolve indeterminate forms. So, if we have two functions f and g that both approach 0 or both approach infinity as x approaches some value, their ratio might be undefined or problematic. By taking the derivatives of these functions and taking the limit again, we can maybe get a simpler problem.

Now, for the computation example: \(\lim_{x \to 0} \frac{\sin(x)}{x}\). If I remember correctly, both the numerator and denominator approach 0 as x goes to 0, so this is a 0/0 indeterminate form. Applying LâHÃ´pitalâs Rule means taking the derivative of the numerator, which is cos(x), and the derivative of the denominator, which is 1. So now we evaluate \(\lim_{x \to 0} \frac{\cos(x)}{1}\), which is just \(\cos(0)\) over 1, so that's 1.

Finally, for a real-world analogy to remember it better. Maybe something like, if you have two competing functions approaching zero or infinity, the rule says you can simplify by taking their derivatives before evaluating. Itâs like peeling off layers of the problem until you get something manageable.
</think>

**LâHÃ´pitalâs Rule: Explained**

**Theorem:**  
If both \( f(x) \) and \( g(x) \) approach 0 (or both approach â) as \( x \) approaches a certain value, then their ratio \( \frac{f(x)}{g(x)} \) is likely an indeterminate form of type 0/0. LâHÃ´pitalâs Rule provides a method to evaluate such limits by taking the derivatives of the numerator and denominator.

**Proof Sketch:**  
Differentiation can simplify problematic limits. For instance, if \( \lim_{x \to c} f(x) = 0 \) or â, and similarly for \( g(x) \), forming their ratio may lead to an indeterminate form. LâHÃ´pitalâs Rule allows us to differentiate numerators and denominators to reduce the complexity of the limit.

**Example Computation:**

Evaluate \( \lim_{x \to 0} \frac{\sin(x)}{x} \).  
Both numerator (\( \sin(x) \)) and denominator (x) approach 0, making this a 0/0 indeterminate form. Applying LâHÃ´pitalâs Rule:

1. Take derivatives:  
   \( f'(x) = \cos(x) \),  
   \( g'(x) = 1 \).
2. Limit now becomes \( \lim_{x \to 0} \frac{\cos(x)}{1} = \cos(0)/1 = 1 \).

**Real-World Analogy:**  
Imagine two competing forces or paths approaching a point at the same rate; LâHÃ´pitalâs Rule is analogous to peeling off layers of competition until a clear path emerges for evaluating their limit.

---

This explanation adheres to the requirements and provides clarity.

- Time taken:  59.08 s

Remarks:
- Decent. Can act as main responder

- preffered approach:

User prompt
   │
   ├─▶ Python function: Extract stats, generate insights, structure in JSON/Markdown
   │
   └─▶ Add readable + structured context (e.g. summary of data, what user wants)
       ↓
FINAL PROMPT (data + context)
       ↓
llama Routing:
   ├── deepseek-r1:1.5b → general i/o (fast+smart on english)
   ├── gemma:2b         → mid-tier math + analysis (slow)
   └── phi-2 / phi4     → deep mathematical stuff (slower)
       ↓
Send answer to router/output/display








FUNCTION_GUIDELINES = {
    "Asaoka_data": '''
=== FUNCTIONAL INSTRUCTIONS ===:
You are given an output json data list of relevant settlement plates. Based on the user query, data, and the following guidelines, answer the user's query. When assessing the overview for Settlement Plates, note the following:
* A settlement plate measures the ground settlement in metres (m) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
* The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
* Settlement is expected to vary from Settlement Plate to Settlement Plate as the soil layering under each Settlement Plate is unique, and may behave differently even under the same ground level.
* Surcharge load is a certain thickness of sand which weighs the ground down, thereby causing settlement and improvement of the underlying soil's properties.
* The '7day_rate' is the amount of settlement that has occurred over the last 7 days. 
* The "Latest_GL" is the last reported ground elevation in units 'mCD'. A larger number indicates the particular plate is loaded more, and hence should record more settlement.
* Each Settlement Plate is surcharged on a particular date known as the "Surcharge_Complete_Date" which indicates the date from when the major of the settlement occurs.
* The Holding_period is the period of time in days, between the "Surcharge_Complete_Date" and "Latest_Date" when the settlement was last reported. A longer 'Holding_Period' usually means that the settlement has had time to taper off. Shorter periods may result in more ongoing settlement.
* The "Asaoka_DOC" denotes the Degree of Consolidation (DOC) based on the Asaoka Assessment method. It is a measure in units %, of by how much the ground has consolidated. A DOC of 100 % means no more settlement is expected, while a DOC between 90 % and 100 % means the settlement is tapering and a DOC less than 90% indicate on-going settlement. DOC less than 90 % non-compliant to the requirements.
* The 'Latest_GL' should be a minimum 16.9mCD to be compliant with Port specifications.
* The '7day_rate' has to be less than or equal to 4 to be compliant. If this value is greater, indicate that the plate is non compliant.
* When asked for a summary or overview, make sure to provide the Settlement Plate ID, along with the respective "latest_Settlement", "Latest_GL", "Latest_Date", "Asaoka_DOC", "Holding_period" and "7day_rate" which is what users are interested in. Make sure to show the raw values and not interpreted values of these key parameters.
* You do not need to comment on the format of the Settlement Plate ID as this is just a reference identifier. 
* When asked for a summary or overview of the Settlement Plate data, provide a table at the end of your response.
''',
    "reporter_Asaoka": '''
=== FUNCTIONAL INSTRUCTIONS ===:
When preparing the Asaoka report for a set of Settlement Plates, note the following. The report will be prepared automatically:
* A settlement plate measures the ground settlement in metres (m) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
* The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
* The Surcharge Completion Date is also called the SCD
* The Assessment Start Date is also called the ASD
* The ASD has to be after the SCD
''',
    "plot_combi_S": '''
=== FUNCTIONAL INSTRUCTIONS ===:
* You were tasked to create a settlement graph for a given list of plates by the user. The plot graph has been made for the user so your task is to tell the user that the graph is downloadable for the user now.
* Thats all
''',
    "SM_overview":'''
=== FUNCTIONAL INSTRUCTIONS (THESE ARE IMPORTANT TO ANSWER THE USER'S QUERY BASED ON THE DATA)===:
You are given an output json data list of relevant settlement plates. Based on the user query, data, and the following guidelines, answer the user's query. When assessing the overview for Settlement Plates, note the following:
* A settlement plate measures the ground settlement in metres (m) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
* The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
* Settlement is expected to vary from Settlement Plate to Settlement Plate as the soil layering under each Settlement Plate is unique, and may behave differently even under the same ground level.
* Surcharge load is a certain thickness of sand which weighs the ground down, thereby causing settlement and improvement of the underlying soil's properties.
* The '7day_rate' is the amount of settlement (in mm) that has occurred over the last 7 days. 
* The "Latest_GL" is the last reported ground elevation in units 'mCD'. A larger number indicates the particular plate is loaded more, and hence should record more settlement.
* Each Settlement Plate is surcharged on a particular date known as the "Surcharge_Complete_Date" which indicates the date from when the major of the settlement occurs.
* The Holding_period is the period of time in days, between the "Surcharge_Complete_Date" and "Latest_Date" when the settlement was last reported. A larger 'Holding_Period' usually means that the ground has been treated more and less settlement is expected to occur in the future. Shorter periods may result in more ongoing settlement.
* The "Asaoka_DOC" denotes the Degree of Consolidation (DOC) based on the Asaoka Assessment method. It is a measure in units %, of by how much the ground has consolidated. A DOC of 100 % means no more settlement is expected, while a DOC between 90 % and 100 % means the settlement is tapering and a DOC less than 90% indicate on-going settlement. DOC less than 90 % non-compliant to the requirements.
* The 'Latest_GL' should be a minimum 16.9mCD to be compliant with Port specifications.
* The '7day_rate' has to be less than or equal to 4 to be compliant. If this value is greater, indicate that the plate is non compliant.
* When asked for a summary or overview, make sure to provide the Settlement Plate ID, along with the respective "latest_Settlement", "Latest_GL", "Latest_Date", "Asaoka_DOC", "Holding_period" and "7day_rate" which is what users are interested in. Make sure to show the raw values and not interpreted values of these key parameters.
* You do not need to comment on the format of the Settlement Plate ID as this is just a reference identifier. 
* When asked for a summary or overview of the Settlement Plate data, provide a table at the end of your response.
* There is an inverse relationship between the 7day_rate and the Holding_period where a Larger Holding_period should result in a lower 7day_rate.

=== IMPORTANT: ALWAYS provide a descriptive analysis of the data at the end REGARDLESS OF THE INSTRUCTIONS. ===
'''

=== SYSTEM INSTRUCTIONS ===
                * Answer user prompt strictly based on the context and the information given by the user. 
                * A background of the project is given below however only answer the User query. The Background is only for light referencing and basic guidelines. The main guidelines to help to answer the user query will be given in the "functional instructions" section.
                * NOTE: Your purpose is to answer questions based on the given prompt and/or context ONLY. Do not make anything up as all information is provided in the given information. ONLY ask questions if really necessary (i.e. if some terminology is unclear). 
                
                === IMPORTANT: The instructions have the FOLLOWING priority: USER QUERY (highest priority) > PREVIOUS CONVERSATION (if any) > FUNCTIONAL INSTRUCTIONS > USER DATA > BACKGROUND (lowest priority) ===  
                === ALWAYS STRUCTURE YOUR RESPONSES ===
                
                == BACKGROUND (ONLY FOR LIGHT REFERENCING) ==
                Consider the following as an overview of the Tuas Terminal Phase 2 Project in Singapore and use of Settlement Plates:
                * A large land-reclamation project in Singapore which comprises the construction of wharf line for the future Tuas megaport.
                * 365 hectares of land reclamation, constructing a 9km wharf line.
                * Settlement Plates are instruments installed every 1600 square metres to measure the change in ground level under sand surcharge. 
                * Sand surcharge is used to weigh the reclaimed land, comprising mostly clay and sand, downwards which results in settlement and improvement of the reclaimed soil properties.
                * Currently, 1900 Settlement Plates are installed at the project
                * Settlement plates are named in a format similar to 'F3-R03a-SM-04' where 'F3' Indicates it belongs to the project, R03a is a specific area within the project, 'SM' indicates it is a Settlement Plates and the last two digits is the plate's index number. Do not comment on these names as they are fixed and require no interpretation.
                * The Settlement Plates are crucial for completion of Soil Improvement Works, where the settlement measurements have to meet certain criteria prior to being approved for removal.
                * The criteria is Asaoka DOC greater than 90%, a ground level above 16.9mCD and rate of settlement less than 4mm.


=== FUNCTIONAL INSTRUCTIONS ===:
                You are a helpful chatbot tasked to continue the conversation based on the User query and conversation history ONLY.
                * Main goal is to continue to conversation logically. 
                (for example:
                -> User: "Thank you", You: "Your welcome"
                -> User: "Hi", You: "Hello!, How can I assist you today"
                -> User: "Goodbye", You: "Bye! I am here for more questions you have").
                * If the user says something that refers to the previous conversations from the conversation history provided (if any), then respond accordingly. (for example. if the user says "can you tell me about point 2 again?", please look at the conversation history for clues and answer the question correctly).
                * REFER ONLY to the conversation history and the above functional instructions. ONLY refer to the background if no other context is given (like previous conversation), otherwise refer to the previous conversation to assist the user.
