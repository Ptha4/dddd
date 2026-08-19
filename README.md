# Mechanistic Interpretability of Semantic Role Assignment in Qwen2.5-0.5B-Instruct

## Abstract

This study investigates how Qwen2.5-0.5B-Instruct represents and uses semantic roles in simple transitive sentences, with particular focus on distinguishing the agent (the entity performing an action) from the patient (the entity receiving the action). The central research question is whether the model develops an internal, syntax-invariant representation of agent and patient roles and, if so, where that information is represented and causally used.

The study uses a controlled synthetic dataset of active and passive sentences, together with several mechanistic-interpretability methods: behavioral evaluation using next-token logit margins, linear probing, representational similarity analysis (RSA), attention analysis, activation patching, and attention-head ablation. The experimental pipeline was redesigned to eliminate several methodological confounds identified in the original implementation, including inconsistent prompt formats, causal-mask effects in attention measurements, lexical imbalance, whole-sequence activation patching, unreliable generated-text scoring, and stale activation caches.

The final results do not support the existence of a simple, syntax-invariant linear representation of semantic role at individual entity-token positions. A probe trained on active sentences achieved 0% accuracy when transferred to passive sentences, with 100% inverted-label accuracy, indicating systematic sensitivity to syntactic position or voice rather than a clean voice-invariant role representation. Similarly, held-out-noun decoding of the forward/reversed binding relation at the final prediction position remained approximately at chance. Representational similarity analysis showed strong lexical similarity throughout the model, with role similarity increasing in later layers but remaining below lexical similarity.

In contrast, activation patching provided causal evidence that late residual-stream states are involved in the model's role-based answer decision. Replacing the final residual representation with that from a role-reversed sentence substantially reduced the correct-answer logit margin from layer 14 onward, with the strongest effects around layers 20–22. Attention-head ablation also affected the answer margin, most notably at layer 15 head 12.

Taken together, the results suggest that Qwen2.5-0.5B-Instruct performs role-sensitive computation in its later layers, but the evidence does not establish a single, linearly decodable, syntax-invariant agent/patient representation at the noun-token level.

## 1. Introduction

Large language models must distinguish semantic relations between entities even when the surface form of a sentence changes. Consider:

"The dog chased the cat."

and

"The cat was chased by the dog."

The same two entities participate in the same event, but their grammatical positions change. In the active sentence, "dog" is the grammatical subject and agent, while "cat" is the patient. In the passive sentence, "cat" occupies the grammatical subject position while "dog" appears in the by-phrase.

This makes active/passive alternations useful for investigating whether a language model represents semantic roles independently of surface syntax.

The present study asks whether Qwen2.5-0.5B-Instruct contains an internal representation that tracks the distinction between agent and patient across these syntactic transformations. More specifically, the study investigates three related questions:

1. Can semantic role information be linearly decoded from entity-token representations?
2. Does the model's internal representation reflect semantic role more strongly than lexical identity?
3. Which internal representations or components causally influence the model's final role-based answer?

The original experimental pipeline contained several analyses intended to answer these questions, including activation extraction, linear probing, representational similarity analysis, attention analysis, activation patching, and attention-head ablation. However, an audit of the original pipeline identified several methodological problems that could produce apparently meaningful results without actually demonstrating a semantic mechanism. The study therefore treats experimental redesign as part of the research methodology rather than simply as implementation debugging.

## 2. Research Design

### 2.1 Model and task

The model investigated was Qwen2.5-0.5B-Instruct. The task was restricted to simple transitive sentences in which two entities participate in an action.

The core contrast was between active and passive constructions. The intended semantic distinction was:

* Agent: entity performing the action.
* Patient: entity receiving the action.

The mechanistic question was whether this semantic distinction persists internally when the syntactic arrangement of the sentence changes.

### 2.2 Dataset design

The final dataset contained 816 examples. It was constructed from:

* 8 training-vocabulary nouns
* 4 held-out-vocabulary nouns
* 6 verbs
* Active and passive constructions
* Both role assignments
* Training and held-out noun splits

The final distribution contained 672 training examples and 144 held-out examples. Each of the four principal conditions contained 204 examples.

The held-out vocabulary was important because it allowed the study to distinguish genuine relational decoding from memorization of particular noun identities.

### 2.3 Counterbalancing

An important issue emerged after the first 816-example dataset was evaluated. Although the four conditions contained equal numbers of examples, the "forward" condition was correlated with which noun appeared first in a pair.

For an unordered pair such as (dog, cat), consistently assigning the first noun as the agent in the forward condition would allow the model to exploit noun position or lexical identity rather than the intended relational structure.

The dataset was therefore revised so that pair orientation alternated across verbs. Three verbs assigned the first entity as the forward agent and three assigned the second entity as the forward agent.

This produced an equal representation of each noun across condition and agent combinations while retaining the total dataset size of 816 examples. The final balance check reported equal counts within the relevant condition-by-agent combinations.

This correction was important because equal numbers of examples across experimental conditions do not by themselves guarantee lexical counterbalancing.

## 3. Experimental Pipeline

A major objective of the redesign was to ensure that all analyses examined the same model computation.

### 3.1 Shared prompt format

The original pipeline used different input representations for different experiments. Activation extraction and attention analysis operated on bare sentences, whereas behavioral evaluation, activation patching, and ablation used chat-formatted prompts.

This created a direct methodological problem: an activation extracted from

"The dog chased the cat."

cannot automatically be assumed to explain behavior generated from a longer instruction-following prompt containing system instructions, a user question, and an answer-generation context.

The revised pipeline therefore introduced a shared prompt-construction utility. Prompt construction, model specification, answer-token lookup, logit-margin calculation, device handling, and dataset selection were standardized across analyses. All 816 examples were aligned to the resulting chat-formatted prompts.

### 3.2 Behavioral measurement

The original behavioral evaluation relied on generated text and substring matching. This could classify an answer as correct simply because the expected noun appeared somewhere in a generated response, even if the response contained both candidate nouns or otherwise expressed an ambiguous answer.

The revised evaluation instead measured the difference between the logits assigned to the two possible answer nouns:

logit(agent) − logit(patient)

A positive margin therefore indicates that the model prefers the correct agent noun over the patient noun.

The answer token was obtained in the actual prompt context rather than by assuming that the first token returned by a standalone tokenizer call represented the answer. This avoided word-boundary and contextual tokenization problems.

This measurement provides a direct and deterministic behavioral variable that is also suitable for causal intervention experiments.

### 3.3 Activation extraction

Residual-stream activations were extracted across all 24 model layers for the 816 aligned examples. The extraction pipeline was subsequently re-run after the final dataset counterbalancing revision.

Activation-cache validation was also introduced because earlier experiments contained mismatches between saved activations and the current dataset. Some historical runs contained only 200 or 600 extracted examples, despite the intended dataset being substantially larger.

The revised pipeline checks that the number of cached activations matches the current aligned dataset before running probing or RSA.

### 3.4 Linear probing

Linear probes were used to test whether role information could be recovered from residual-stream representations.

The most important test was cross-syntactic transfer:

* Train the probe on active sentences.
* Test it on passive sentences.
* Label the entity representing the agent as one class.
* Label the patient as the other class.

If a common semantic-role direction existed at entity-token states, a probe trained on active sentences should ideally transfer to passive sentences despite the change in syntax.

However, this design also has an important vulnerability: active and passive sentences simultaneously change grammatical position, word order, distance from the prediction site, and voice morphology. Therefore, poor transfer cannot by itself be interpreted as evidence that the model lacks semantic-role information.

The final analysis consequently reported both ordinary accuracy and inverted-label accuracy rather than treating 0% accuracy as simple failure.

### 3.5 Representational similarity analysis

RSA was used to compare the similarity structure of entity representations.

The key comparison was between:

* lexical similarity: whether representations are similar because they correspond to the same noun;
* role similarity: whether representations are similar because they occupy the same semantic role.

This analysis tests whether the geometry of the residual stream becomes increasingly organized around semantic role as processing proceeds.

### 3.6 Attention analysis

The original attention analysis attempted to measure attention from the verb token to the agent and patient.

This was invalid as a semantic-role measure in a causal decoder because attention is restricted to preceding tokens. In active sentences, the patient occurs after the verb, so the verb cannot attend to it. In passive sentences, the agent often occurs after the verb and is similarly inaccessible.

Consequently, an apparent agent/patient attention difference could be produced entirely by causal masking and word order.

The revised analysis therefore examined attention from the final prediction-site token, which can attend to both entities and is directly involved in the next-token answer computation.

The resulting attention analysis was treated as descriptive rather than causal. Candidate heads were then passed to ablation experiments for causal testing.

### 3.7 Activation patching

Activation patching was used to test whether particular residual representations causally affect the model's answer.

Rather than replacing the entire sequence of residual activations, the revised approach focused on the final prediction-token residual state. The intervention substituted the representation associated with a role-reversed sentence into the target computation.

The behavioral consequence was measured using the controlled agent-versus-patient logit margin.

This creates a direct causal test:

If replacing a representation from a role-reversed example changes the answer preference, then that representation contains information that is causally relevant to the model's decision.

### 3.8 Attention-head ablation

Attention analysis was used to identify candidate heads, but candidate selection alone was not interpreted as evidence of mechanism.

The candidate heads were subsequently ablated and the effect on the answer logit margin was measured. This separates descriptive attention patterns from causal component-level evidence.

## 4. Results

### 4.1 Behavioral evaluation

The first 816-example evaluation, conducted before the final lexical counterbalancing correction, produced substantial asymmetries between forward and reversed conditions.

For the held-out vocabulary, accuracy was:

| Condition         | Accuracy | Mean logit margin |
| ----------------- | -------: | ----------------: |
| Active, forward   |    83.3% |            +3.688 |
| Active, reversed  |    97.2% |            +4.069 |
| Passive, forward  |    25.0% |            −0.756 |
| Passive, reversed |    38.9% |            −0.360 |

For the training vocabulary:

| Condition         | Accuracy | Mean logit margin |
| ----------------- | -------: | ----------------: |
| Active, forward   |    53.0% |            +0.052 |
| Active, reversed  |    98.2% |            +4.884 |
| Passive, forward  |    48.8% |            −0.571 |
| Passive, reversed |    90.5% |            +4.725 |

These results revealed a major asymmetry between forward and reversed conditions. Importantly, the subsequent investigation showed that the first 816-example dataset still contained a lexical-condition confound. Therefore, these numbers are not treated as the final behavioral result. Instead, they served as evidence motivating the final counterbalancing revision.

This is an important methodological result: the controlled logit-margin evaluation did not merely produce a cleaner metric; it exposed an imbalance that could have been mistaken for a property of the model.

### 4.2 Cross-syntactic role probing

The most direct test of a syntax-invariant role representation produced a striking result.

A linear probe trained on active-sentence entity representations achieved:

* 0% accuracy when evaluated using the original agent/patient labels on passive sentences.
* 100% accuracy when the passive labels were inverted.

The complete inversion occurred across layers rather than appearing as an isolated effect.

This result indicates that the probe was not simply failing to extract useful information. Instead, the representation contained a highly systematic distinction that reversed under the active-to-passive transformation.

However, this distinction cannot safely be identified with semantic role. Active and passive constructions change several correlated variables simultaneously, including word position, grammatical subjecthood, distance to the answer position, and voice morphology.

Therefore, the result is evidence against a simple voice-invariant linear role direction at individual noun-token states, but it is not evidence that the model does not understand semantic roles.

### 4.3 Held-out-noun binding probe

A second probing analysis examined whether the model represented the forward/reversed binding relation at the final prompt token in a way that generalized to nouns not seen during training.

The held-out-noun linear decoding result was approximately chance.

This is significant because a high-performing probe in this setting could have indicated a representation of the relation itself rather than a representation tied to specific lexical identities.

The near-chance result therefore provides no evidence for a robust, linearly decodable forward/reversed binding representation at the final prediction token.

### 4.4 Representational similarity

RSA showed strong lexical similarity throughout the model.

Role similarity became stronger in later layers, suggesting that the model's representations become increasingly organized according to the relational structure relevant to the task.

However, role similarity did not exceed lexical similarity.

The result therefore suggests a late increase in role-related organization without demonstrating that semantic role becomes the dominant organizing factor in the representation space.

### 4.5 Activation patching

Activation patching produced the strongest causal evidence in the study.

When a representation from a role-reversed sentence was introduced into the final residual-stream state, the model's correct-answer logit margin decreased substantially from approximately layer 14 onward.

The largest effects occurred around layers 20–22.

This result indicates that late residual representations are causally involved in the model's role-sensitive answer computation.

The interpretation is stronger than a simple correlation: the representation was actively replaced, and the model's answer preference changed as a consequence.

However, the result does not identify a specific semantic-role feature or demonstrate that the patched state contains a clean agent/patient variable. It establishes causal involvement of late computation, not the exact content of the representation.

### 4.6 Attention-head ablation

Attention-head ablation also produced measurable effects on the answer margin.

Among the tested components, layer 15 head 12 showed a particularly strong effect.

The broader attention analysis identified several candidate heads based on final-token attention differences, including:

* L19.H2
* L20.H12
* L23.H12
* L20.H7
* L17.H8

These heads were treated as candidates rather than as established semantic-role mechanisms. Their relevance was evaluated through ablation rather than inferred directly from their attention patterns.

## 5. Interpretation

The results provide evidence for a distinction between representational decoding and causal computation.

At the entity-token level, the experiments do not reveal a clean semantic-role representation that is invariant to syntax and easily recovered by a linear classifier. The active-to-passive probe completely inverted its predictions, and the held-out-noun binding probe remained approximately at chance.

This does not imply that the model lacks the ability to distinguish agent from patient. The behavioral task and causal intervention results indicate that role-sensitive computation does occur.

Instead, the evidence suggests that the relevant information may be distributed across multiple representations or computed dynamically through later layers rather than being stored as a simple, syntax-independent feature at the noun token.

The RSA result is consistent with this interpretation. Lexical information remains strong throughout the network, while role-related similarity increases in later layers without overtaking lexical similarity.

Activation patching provides the clearest evidence for where the relevant computation becomes causally important. The strong effects beginning around layer 14 and peaking around layers 20–22 indicate that the final stages of processing are important for determining which entity should receive the answer label.

Thus, the most defensible mechanistic interpretation is not:

"The model stores agenthood as a single invariant feature."

Rather, it is:

"The model's late residual computation contains information that is causally important for resolving the agent/patient distinction, but the experiments do not establish a simple, syntax-invariant representation of that distinction at individual entity-token states."

## 6. Methodological Contributions

An important contribution of the study is methodological.

Several apparent mechanistic findings in the original pipeline could have been misleading because the experimental measurements did not correspond cleanly to the computation being investigated.

The redesigned pipeline addressed these issues by:

1. Standardizing the prompt format across all analyses.
2. Aligning entity tokens within the complete chat prompt.
3. Replacing free-form substring scoring with controlled answer-token logit margins.
4. Counterbalancing lexical identities across experimental conditions.
5. Validating activation-cache sizes before probing and RSA.
6. Replacing causal-mask-confounded verb attention with final-token attention.
7. Treating attention patterns as candidate evidence rather than causal evidence.
8. Replacing whole-sequence activation patching with targeted final-token interventions.
9. Dynamically transferring descriptive attention candidates into ablation.
10. Reporting inverted-label probe performance rather than interpreting 0% accuracy in isolation.

These changes substantially narrow the range of conclusions that can legitimately be drawn from the experiments.

## 7. Limitations

The central limitation is that the active/passive manipulation is not a pure test of semantic role invariance. Changing from active to passive simultaneously changes syntax, word order, grammatical subjecthood, positional information, distance between entities and the prediction token, and morphological structure.

Consequently, the 0%/100% probe result cannot distinguish among all of these possible sources of systematic inversion.

A second limitation is that the dataset is synthetic and relatively small. The 816-example design was selected partly to remain computationally manageable on the available hardware. While lexical holdout and counterbalancing improve experimental control, the results should not automatically be generalized to naturalistic language.

A third limitation is that activation-cache validation currently checks the number of examples but does not fully establish dataset provenance. A more rigorous pipeline would associate every cache with a dataset hash, model revision, tokenizer revision, prompt version, and extraction configuration.

A fourth limitation concerns attention. Even after correcting the causal-mask problem, attention remains a descriptive quantity. High attention to an entity does not establish that the head is causally responsible for the model's decision. The ablation experiments address this partially, but ablation effects themselves do not necessarily reveal the precise computation performed by a head.

Finally, the final source material records the redesign and principal results but does not provide a complete quantitative account of every statistical comparison or intervention effect. The conclusions should therefore remain at the level supported by the reported results.

## 8. Conclusion

This study examined whether Qwen2.5-0.5B-Instruct represents and uses agent/patient distinctions in a syntax-invariant manner.

The results do not support a simple, linearly decodable semantic-role representation at individual noun-token residual states. Cross-syntactic role probing produced systematic label inversion, while held-out-noun decoding at the final prediction position remained approximately at chance. RSA showed that lexical similarity remains stronger than role similarity, although role-related structure increases in later layers.

At the same time, activation patching demonstrated substantial causal sensitivity in the late residual stream. Replacing the final residual representation with that from a role-reversed sentence reduced the correct-answer logit margin from layer 14 onward, with the strongest effects around layers 20–22. Attention-head ablation further showed that specific late components can influence the model's answer margin.

The resulting picture is therefore one of late, distributed role-sensitive computation rather than a single invariant semantic-role feature.

The strongest conclusion supported by the experiments is that late residual representations causally contribute to the model's agent/patient decision, while the existence of a syntax-invariant, linearly decodable agent/patient representation at individual entity-token positions remains unsupported.

This distinction is important. The study does not show that the model lacks semantic-role representations. Rather, it shows that the particular representation hypothesized at the beginning of the experiment—a simple role-invariant direction recoverable from individual entity-token activations—was not found. The causal evidence instead points toward a more distributed computation emerging in the later layers of the network.
