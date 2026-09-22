"""
================================================================================
RKMJ-Core Training Dataset & Corpus Generator
================================================================================
Location: TEST/training/dataset.py
Contains:
1. 50 curated, grammatically rigorous domain-expert paragraphs spanning Natural
   Sciences, Mathematics, World History, Logic, and Philosophy.
2. Dataset loader supporting Bingsu/openwebtext_20p with streaming and chunking.
3. PyTorch Causal Language Modeling Dataset interface (x -> y next-token prediction).
================================================================================
"""

import os
from typing import Iterator, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, IterableDataset


EXPERT_PARAGRAPHS = [
    # --- NATURAL SCIENCES (Physics, Chemistry, Molecular Biology, Earth Systems, Astrophysics) ---
    "Photosynthesis converts electromagnetic solar radiation into stable chemical potential energy within organic carbohydrates. The light-dependent reactions occur across the thylakoid membranes of chloroplasts, where photon absorption initiates the photolysis of water molecules. This catalytic dissociation produces gaseous oxygen, protons, and high-energy electrons that traverse an organized electron transport chain. The resulting electrochemical proton gradient drives adenosine triphosphate synthase to produce adenosine triphosphate and reduced nicotinamide adenine dinucleotide phosphate. These energetic intermediates subsequently fuel the Calvin cycle within the stroma, wherein carbon dioxide undergoes enzymatic fixation by ribulose-1,5-bisphosphate carboxylase-oxygenase to synthesize three-carbon phosphoglycerate precursors.",
    "Cellular respiration oxidizes hexose sugars through a sequence of convergent metabolic pathways to generate universal biochemical energy. Glycolysis initiates the enzymatic cleavage of glucose into two pyruvate molecules within the aqueous cytoplasm, yielding a net gain of two adenosine triphosphate molecules and two molecules of reduced nicotinamide adenine dinucleotide. Under aerobic conditions, pyruvate translocates across the double mitochondrial membrane into the matrix, where the pyruvate dehydrogenase complex oxidatively decarboxylates it into acetyl coenzyme A. Acetyl groups subsequently enter the tricarboxylic acid cycle, undergoing cyclical oxidation to release carbon dioxide while reducing flavin and nicotinamide electron carriers. The high-energy electrons donated to the inner mitochondrial electron transport chain drive oxidative phosphorylation, establishing the transmembrane proton motive force that synthesizes the preponderance of cellular adenosine triphosphate.",
    "Quantum electrodynamics describes the fundamental relativistic interactions between charged leptonic matter and the electromagnetic gauge field. Electrons and positrons exchange virtual photons, which serve as the gauge bosons mediating electromagnetic force according to the Abelian symmetry group U(1). Renormalization techniques eliminate non-physical infinite divergences that emerge from higher-order loop Feynman diagrams, yielding extraordinarily precise predictions for physical observables. The anomalous magnetic dipole moment of the muon provides a stringent experimental benchmark for the mathematical consistency of this gauge theory. Precision measurements of the fine-structure constant confirm that quantum electrodynamics remains among the most rigorously verified frameworks in fundamental physics.",
    "General relativity reformulates gravitation as the intrinsic geometric curvature of four-dimensional pseudo-Riemannian spacetime rather than an instantaneous Newtonian force. Mass and energy distributions dictate the metric tensor of spacetime through Einstein field equations, which equate the Einstein curvature tensor to the stress-energy-momentum tensor of matter. In the absence of non-gravitational external forces, test particles and photons traverse geometric geodesics that represent the straightest possible trajectories across curved spacetime. Massive compact bodies, such as neutron stars and black holes, induce profound local spacetime distortion, leading to measurable gravitational time dilation and the deflection of light rays. The catastrophic coalescence of binary black hole systems radiates ripples through spacetime known as gravitational waves, propagating outward at the invariant speed of light.",
    "Enzymatic catalysis accelerates biological chemical transformations by substantially lowering the Gibbs free energy barrier of the transition state. The active site of a globular enzyme possesses a geometrically precise spatial arrangement of amino acid residues that stereospecifically bind the designated substrate. Induced fit mechanisms optimize complementary non-covalent interactions, including electrostatic bonds, hydrogen bonding, and transient hydrophobic contacts. Acid-base and covalent catalytic pathways stabilize the transition-state intermediate, preventing excessive thermodynamic kinetic resistance. Consequently, intracellular metabolic reactions achieve kinetic rate enhancements exceeding a million-fold relative to uncatalyzed reactions under physiological conditions.",
    "Thermodynamics dictates the macroscopic directional evolution of physical systems through conservation laws and statistical entropy constraints. The first law establishes that total internal energy remains invariant in an isolated system, undergoing transformation exclusively through work and heat exchange. The second law mandates that the total thermodynamic entropy of an isolated system must increase or remain constant during any spontaneous thermodynamic transformation. Reversible processes represent idealized thermodynamic trajectories where entropy generation remains zero, maximizing thermal efficiency according to Carnot theorem. Irreversible processes generate positive entropy, establishing an asymmetric arrow of time across macroscopic physical phenomena.",
    "The central dogma of molecular biology delineates the directional flow of genetic sequential information from nucleic acid templates to functional polypeptide structures. Deoxyribonucleic acid undergoes replication catalyzed by DNA polymerases, maintaining informational fidelity through complementary base-pairing and exonucleolytic proofreading mechanisms. Transcription factors recruit RNA polymerase to specific genomic promoter regions, transcribing genetic code into messenger ribonucleic acid. Ribosomes read triplet nucleotide codons along the messenger RNA during translation, orchestrating the condensation of amino acids into elongating polypeptide chains. Transfer RNA molecules serve as adaptor complexes, ensuring the precise stereochemical translation of genetic instructions into catalytic proteins.",
    "Stellar nucleosynthesis generates the chemical elements through successive nuclear fusion regimes occurring within the pressurized cores of massive stars. Main sequence stars sustain hydrostatic equilibrium by fusing hydrogen nuclei into helium via the proton-proton chain and the carbon-nitrogen-oxygen catalytic cycle. As core hydrogen exhausts, gravitational contraction elevates core temperature and density, triggering helium fusion into carbon and oxygen through the triple-alpha process. Advanced evolutionary stages in massive stars sequentially ignite carbon, neon, oxygen, and silicon fusion burning shells, culminating in an iron-peak core that cannot undergo exothermic fusion. The subsequent core collapse triggers a core-collapse supernova, dispersing synthesized heavy elements throughout the interstellar medium via explosive nucleosynthesis.",
    "Plate tectonics governs the large-scale kinematic displacement and dynamic recycling of the lithosphere across the convective asthenosphere. Convection currents within the silicate mantle drive divergent boundaries where seafloor spreading creates new basaltic oceanic crust along mid-ocean ridge systems. Convergent plate margins produce subduction zones, forcing denser oceanic lithosphere beneath continental margins and generating deep oceanic trenches alongside volcanic island arcs. Transform faults accommodate horizontal strike-slip displacement between adjacent tectonic plates, periodically releasing accumulated elastic strain as seismic earthquake ruptures. This continuous lithospheric recycling regulates the global geochemical carbon cycle over planetary geological epochs.",
    "Chemical equilibrium characterizes a dynamic state wherein the forward and reverse reaction rates of a closed chemical system become precisely equal. The law of mass action defines the equilibrium constant as the ratio of product activities to reactant activities raised to their respective stoichiometric coefficients. Le Chatelier principle dictates that a system at equilibrium responds to external perturbations in temperature, pressure, or concentration by shifting its state to partially counteract the disturbance. Exothermic reactions exhibit a decrease in the equilibrium constant when thermal energy increases, shifting equilibrium toward the reactants. Conversely, pressure alterations shift gaseous equilibria toward the side possessing fewer moles of gaseous constituents.",

    # --- MATHEMATICS (Linear Algebra, Real Analysis, Number Theory, Topology, Probability) ---
    "Linear algebra analyzes the geometric and algebraic properties of vector spaces, linear transformations, and finite-dimensional matrices. A linear transformation maps vectors between vector spaces while rigorously preserving vector addition and scalar multiplication operations. The spectral theorem guarantees that every real symmetric matrix possesses an orthogonal basis of eigenvectors associated with purely real eigenvalues. Matrix diagonalization decomposes a linear operator into canonical diagonal form, revealing invariant dimensional invariant directions and simplifying powers of matrices. Singular value decomposition generalizes this factorization to arbitrary rectangular matrices, providing fundamental computational foundations for low-rank approximation, dimensionality reduction, and modern numerical optimization.",
    "Calculus formalizes the continuous variation and geometric accumulation of quantities through differential and integral operations. The derivative quantifies the instantaneous rate of change of a differentiable function as the limiting ratio of functional increments to input increments. The Riemann integral aggregates infini-tesimal area partitions beneath a curve, formalizing continuous summation across bounded domains. The fundamental theorem of calculus unifies these dual branches, establishing that differentiation and definite integration serve as mutually inverse operations. Taylor series expansions further approximate sufficiently smooth analytic functions as infinite polynomial series centered around a localized point.",
    "Number theory investigates the arithmetic properties and structural distributions of integers, prime numbers, and rational relationships. The fundamental theorem of arithmetic asserts that every integer greater than one admits a unique prime factorization up to the permutation of factors. Modular arithmetic defines equivalence relations across integers based on divisibility by a modulus, underpinning contemporary public-key cryptography. The distribution of prime numbers asymptotically obeys the prime number theorem, maintaining an intimate analytical connection with the non-trivial zeros of the Riemann zeta function. Diophantine equations examine whether polynomial equations admit purely integer solutions, historically driving profound developments in algebraic geometry.",
    "Topology investigates the geometric and spatial invariants that remain preserved under continuous deformations such as stretching, twisting, and crumpling without tearing. A topological space defines neighborhood structures axiomatically through open sets without requiring an explicit metric distance function. Homeomorphisms establish topological equivalence between spaces whenever a continuous bijective mapping possesses a continuous inverse. Compactness generalizes the property of closed and bounded Euclidean subsets, ensuring that every open cover admits a finite subcover. Fundamental groups and homology groups extract algebraic invariants that distinguish topologically non-equivalent manifolds across higher dimensions.",
    "Probability theory formulates mathematical measures for uncertainty through Kolmogorov axiomatic framework defined over sample spaces, sigma-algebras, and probability measures. Random variables map stochastic outcomes to real-valued measurable spaces, characterized by cumulative distribution functions and probability density functions. The law of large numbers guarantees that the empirical mean of independent and identically distributed random variables converges almost surely to their theoretical expected value. The central limit theorem establishes that normalized sums of independent random variables converge in distribution to a Gaussian normal distribution regardless of the underlying distribution shape. Conditional expectation serves as the cornerstone for martingale theory and continuous-time stochastic processes.",
    "Complex analysis investigates functions of complex variables that possess complex differentiability across open subsets of the complex plane. A complex function is holomorphic if its derivative exists everywhere within an open domain, which implies that it satisfies the Cauchy-Riemann differential equations. Holomorphic functions are infinitely differentiable and analytic, meaning they coincide locally with their convergent power series expansions. Cauchy integral theorem establishes that the contour integral of a holomorphic function along any simple closed path in a simply connected domain evaluates identically to zero. The residue theorem extends this contour integration to meromorphic functions with isolated singularities, enabling the exact analytical evaluation of challenging real integrals.",
    "Abstract algebra studies axiomatic algebraic structures, primarily groups, rings, fields, and modules over rings. A group formalizes the concept of symmetry by coupling a set with an associative binary operation possessing an identity element and universal inverses. Lagrange theorem states that the order of any subgroup divides the order of a finite parent group, constraining permissible substructures. Ring theory investigates systems endowed with two compatible binary operations, generalizing arithmetic concepts such as ideals, quotient rings, and unique factorization domains. Galois theory bridges field theory and group theory, demonstrating that the roots of polynomial equations of degree five or higher cannot be solved by radicals.",
    "Differential geometry applies calculus and linear algebra to analyze smooth curves, surfaces, and differentiable manifolds. A Riemannian metric assigns an inner product to the tangent space at each point of a manifold, allowing the rigorous measurement of lengths, angles, and volumes. The Levi-Civita connection provides an intrinsic covariant derivative that parallel-transports tangent vectors along smooth curves without torsional deviation. Riemann curvature tensor encapsulates the failure of covariant derivatives to commute, quantifying intrinsic geometric curvature independent of any embedding space. Geodesics generalize straight lines to curved manifolds, minimizing arc length between proximate points according to the calculus of variations.",
    "Discrete mathematics explores non-continuous mathematical structures, incorporating graph theory, combinatorics, and algorithmic complexity. A graph consists of a set of vertices linked by edges, serving as an abstract model for relational networks and pairwise connections. Euler characteristic provides an invariant topological relation for planar graphs, linking vertices, edges, and bounded faces through a constant formula. Combinatorics develops rigorous counting techniques, utilizing generating functions and recurrence relations to enumerate structured configurations. Graph traversal algorithms, such as breadth-first search and depth-first search, determine path connectivity and optimal routing across computational networks.",
    "Measure theory provides the rigorous analytical foundation for integration, advanced probability, and functional analysis. A sigma-algebra defines a collection of subsets that remains closed under complementation and countable unions, specifying measurable events. The Lebesgue measure generalizes classical notions of length, area, and volume to intricate subsets of Euclidean space, accommodating discontinuous limits. The Lebesgue dominated convergence theorem establishes conditions under which integration and limits may be interchanged for sequences of measurable functions. Radon-Nikodym theorem formalizes the derivative of one measure with respect to another, establishing the foundation for conditional probability and information divergence.",

    # --- WORLD HISTORY (Antiquity, Medieval Transitions, Renaissance, Modernity) ---
    "The Neolithic Revolution transitioned human societies from nomadic hunting and gathering to sedentary agricultural civilization. The deliberate domestication of cereal grains such as emmer wheat and barley fostered permanent settlements across the fertile crescent of southwest Asia. Increased caloric yields generated agricultural surpluses, which catalyzed demographic population expansion and occupational specialization. Craft production, metallurgy, and centralized administrative hierarchies emerged to manage surplus distribution and territorial boundaries. This structural societal transformation established the fundamental prerequisite conditions for early urbanism and monumental architecture.",
    "Ancient Mesopotamian civilizations established the earliest systematic urban centers, codified legal structures, and written bureaucratic records. The development of cuneiform script on clay tablets enabled Sumerian city-states to document agricultural inventories, commercial transactions, and administrative decrees. The Code of Hammurabi codified legal jurisprudence in Babylon, establishing reciprocal retaliatory penalties and formalizing contract obligations across social strata. Irrigation canals diverted the waters of the Tigris and Euphrates rivers, sustaining intensive agriculture despite arid regional conditions. Persistent competition among rival city-states eventually gave rise to centralized territorial empires across the Mesopotamian basin.",
    "Classical Athenian democracy cultivated citizen participation in direct political governance and legislative deliberation. The reforms of Cleisthenes reorganized civic tribes to diminish aristocratic factional dominance, empowering the popular assembly known as the Ekklesia. Citizens voted directly on legislation, military deployments, and executive policies through majority consensus without representative mediation. Legal trials were conducted before large juries selected by lot from the citizen body, mitigating systemic judicial bribery. However, political franchise excluded enslaved populations, resident foreigners, and women, confining democratic participation to adult male citizens.",
    "The Roman Republic established a mixed constitution that combined monarchical, aristocratic, and democratic administrative elements. Executive authority was vested in two annually elected consuls who commanded military legions and executed senate decrees. The Roman Senate, composed predominantly of patrician elites, controlled fiscal expenditures and foreign policy directives. Plebeian citizens secured legislative influence through the creation of the Tribunate, which possessed the legal authority to veto magistrate actions. Internal socioeconomic polarization and military reorganization during the late republic eventually precipitated civil wars that dismantled republican governance and established the Principate.",
    "The Silk Road constituted an expansive trans-Eurasian network of commercial trade routes connecting China, Central Asia, the Middle East, and Mediterranean Europe. Caravans transported high-value commodities such as silk, spices, precious metals, and porcelain across hazardous terrestrial terrains. Beyond mercantile commerce, the routes facilitated the cross-cultural transmission of technologies, including papermaking and gunpowder manufacturing. Religious traditions, notably Buddhism, Nestorian Christianity, and Islam, diffused along trade arteries, transforming regional cultures across Eurasia. This sustained commercial interregional connectivity stimulated diplomatic embassies and geographic exploration between distant civilizations.",
    "The Islamic Golden Age fostered substantial intellectual advancements in mathematics, astronomy, medicine, and philosophy between the eighth and fourteenth centuries. The Abbasid caliphate established the House of Wisdom in Baghdad, sponsoring the translation of Greek, Persian, and Indian scientific manuscripts into Arabic. Scholars such as Al-Khwarizmi formulated the foundational principles of symbolic algebra and popularized positional decimal numeral systems. Ibn al-Haytham revolutionized experimental optics through empirical investigations of light refraction and camera obscura mechanics. Ibn Sina codified contemporary medical knowledge in The Canon of Medicine, which served as a standard pedagogical authority across European and Asian universities for centuries.",
    "The Black Death of the fourteenth century decimated European and Mediterranean populations, triggering profound socio-economic restructuring. The rapid spread of Yersinia pestis via rodent flea vectors caused acute demographic collapse, killing an estimated third of Europe population. Severe labor shortages diminished feudal authority, granting surviving agrarian laborers significant leverage to demand monetary wages and reduced manorial obligations. Traditional religious authorities experienced reputational erosion as ecclesiastical institutions failed to provide spiritual or physical relief. The resulting economic realignment facilitated the decline of serfdom and catalyzed early wage-labor agricultural systems across Western Europe.",
    "The Renaissance ignited a profound cultural, artistic, and philosophical revival across Western Europe originating in Italian city-states. Humanist scholars rediscovered and translated classical Greco-Roman manuscripts, emphasizing secular literary critique, rhetoric, and moral philosophy. Linear perspective, anatomical realism, and chiaroscuro techniques revolutionized visual representation in paintings and architectural design. Merchant banking families, such as the Medici in Florence, channeled commercial capital into civic and religious artistic patronage. The invention of the movable-type printing press accelerated the democratization of literacy, undermining centralized scholastic orthodoxy.",
    "The Scientific Revolution transformed epistemology by establishing empirical observation, controlled experimentation, and mathematical formulation as the standard criteria for scientific truth. Nicolaus Copernicus formulated a mathematically coherent heliocentric astronomical model, challenging geocentric Ptolemaic dogma. Galileo Galilei utilized optical telescopes to observe lunar topography, Jovian moons, and planetary phases, providing observational corroboration for heliocentrism. Johannes Kepler derived mathematical laws of planetary motion, demonstrating that celestial orbits form ellipses rather than uniform circular paths. Isaac Newton integrated these discoveries into the universal law of gravitation and classical mechanics, unifying terrestrial and celestial physics.",
    "The Industrial Revolution fundamentally transformed global production by replacing manual artisanal labor with mechanized manufacturing systems powered by fossil fuels. The invention of the atmospheric steam engine facilitated localized power generation, decoupling industrial manufacturing from water wheels and biological muscle. Mechanized textile factories concentrated labor into urban industrial hubs, generating rapid urbanization and altering traditional domestic social structures. The expansion of railway networks dramatically reduced overland transportation costs, integrating domestic and continental commodity markets. This economic intensification established industrial capitalism as the dominant macroeconomic system across nineteenth-century global trade.",

    # --- LOGIC & COMPUTATION (Propositional Logic, Computability, Automata, Information Theory) ---
    "Propositional logic formalizes deductive reasoning through declarative statements evaluated under bivalent truth functional semantics. Atomic propositions combine through logical connectives, including conjunction, disjunction, implication, and negation, to construct compound formulas. Truth tables systematically enumerate all valuation assignments, verifying whether a given proposition constitutes a tautology, contradiction, or contingent sentence. Inference rules such as modus ponens and modus tollens establish validity by preserving truth value through formal deductive deductions. The sound and complete character of propositional logic guarantees that all syntactically derivable theorems correspond directly to semantic tautologies.",
    "First-order predicate logic extends propositional syntax by incorporating individual constants, variables, predicates, and universal or existential quantifiers. Predicates represent properties or relations across a designated domain of discourse, enabling fine-grained formalization of mathematical assertions. Gödel completeness theorem establishes that every semantically valid first-order formula possesses a formal finite deductive proof within standard deductive calculi. Conversely, Gödel first incompleteness theorem proves that any consistent formal axiomatic system capable of formulating elementary arithmetic contains undecidable statements. Consequently, no consistent recursive axiomatic system can achieve absolute mathematical completeness while formalizing its own consistency.",
    "Computability theory establishes the theoretical mathematical limits of algorithmic problem solving through formal models of computation. Alan Turing formulated the Turing machine, demonstrating that an abstract state machine manipulating a linear tape accurately captures mechanical computation. The Church-Turing thesis conjectures that any physically realizable algorithmic process can be simulated by a universal Turing machine. The halting problem establishes that no general algorithm can decide whether an arbitrary program will eventually terminate or execute infinitely. This fundamental undecidability demonstrates that intrinsic algorithmic boundaries exist independently of physical computational power.",
    "Computational complexity theory categorizes computational problems according to the asymptotic resources required for their algorithmic resolution. The complexity class P encompasses decision problems solvable by a deterministic Turing machine within polynomial time bounds. The class NP comprises decision problems whose positive solutions admit verification by a deterministic algorithm in polynomial time. The P versus NP problem questions whether every efficiently verifiable problem can also be efficiently solved from scratch. Cook-Levin theorem proved that the Boolean satisfiability problem is NP-complete, meaning a polynomial-time solution for SAT would prove that P equals NP.",
    "Automata theory models discrete computational systems using mathematical abstractions possessing finite, bounded, or unbounded memory configurations. Finite state machines process input symbols by transitioning among a predetermined set of internal states, recognizing regular languages. Pushdown automata augment finite controllers with a last-in-first-out stack memory, enabling the recognition of context-free languages. The Chomsky hierarchy systematically organizes formal grammars into regular, context-free, context-sensitive, and recursively enumerable language classes. Pumping lemmas provide formal contradiction mechanisms for proving that specific formal languages transcend the expressive capacity of simpler automata classes.",
    "Information theory formalizes the quantification, storage, and reliable communication of discrete data across noisy communication channels. Claude Shannon defined information entropy as the expected value of the self-information contained within stochastic message outcomes. Shannon noiseless source coding theorem establishes the fundamental limit for lossless data compression as the entropy rate of the source. The noisy-channel coding theorem guarantees that information can be transmitted with arbitrarily low error rates provided the transmission rate remains below channel capacity. Differential entropy extends these discrete probabilistic measures to continuous probability density functions in signal processing.",
    "Lambda calculus formalizes computation through pure variable substitution, functional abstraction, and function application without relying on stateful memory registers. Church-Rosser theorem establishes the confluence property, guaranteeing that reduction order does not alter the unique normal form of an expression. Alpha-conversion provides systematic variable renaming to avoid identifier capture, while beta-reduction executes evaluation through argument substitution. Combinatory logic demonstrates that variables can be entirely eliminated using fixed functional combinators such as S, K, and I. The untyped lambda calculus possesses universal Turing completeness, serving as the mathematical bedrock for functional programming paradigms.",
    "Type theory classifies mathematical terms into structured types to prevent logical paradoxes and ensure syntactic semantic consistency. Simply typed lambda calculus equips functional expressions with explicit types, guaranteeing strong normalization where all well-typed terms terminate. The Curry-Howard correspondence establishes a deep isomorphism linking formal logical systems with type theories, equating propositions to types and proofs to executable programs. Dependent type theory allows types to depend on values, enabling the unified formalization of mathematical theorems and software verification within interactive theorem provers. Constructive Martin-Löf type theory forms the conceptual core for modern intuitionistic mathematics and proof assistants.",
    "Modal logic enriches classical logic by formalizing modes of truth, including necessity, possibility, epistemic belief, and temporal progression. Kripke relational semantics evaluates modal propositions across structured graphs of accessible possible worlds. A proposition is necessary if it evaluates as true in every accessible possible world, whereas it is possible if it holds in at least one accessible state. Modal system S5 stipulates that the accessibility relation constitutes an equivalence relation, rendering iterated modal operators reducible. Temporal and epistemic variants of modal logic provide powerful formal frameworks for verifying concurrent software protocols and multi-agent distributed systems.",
    "Boolean algebra provides the algebraic substrate for digital logic design, circuit optimization, and computer microprocessor architecture. The algebraic operations of conjunction, disjunction, and complementation map directly onto physical semiconductor logic gates such as AND, OR, and NOT. De Morgan laws provide transformation identities that allow any Boolean expression to be rewritten using universal NAND or NOR gate logic. Karnaugh maps and the Quine-McCluskey algorithm algorithmically minimize Boolean functions to reduce hardware gate counts and propagation delays. These foundational algebraic identities enable binary arithmetic units to execute high-throughput integer and floating-point computations.",

    # --- PHILOSOPHY (Epistemology, Ethics, Metaphysics, Philosophy of Mind) ---
    "Epistemology investigates the fundamental nature, acquisition, and normative justification of human knowledge and rational belief. Classical analysis defines knowledge as justified true belief, demanding that an agent hold an epistemically warrantable true proposition. The Gettier problem challenged this tripartite definition by presenting scenarios where justified true beliefs arise through epistemic luck. Foundationalism posits that knowledge rests upon self-justifying basic beliefs that require no further inferential support. Conversely, coherentism argues that justification stems from the mutual consistency and systemic coherence of an interconnected web of beliefs.",
    "Deontological ethics evaluates the moral status of human actions based on adherence to universal duties rather than empirical consequences. Immanuel Kant formulated the categorical imperative, demanding that agents act only on maxims they can simultaneously will as universal laws. The principle of humanity dictates that rational beings must never be treated merely as a means to an end, but always as intrinsic ends in themselves. Deontological frameworks distinguish moral duty from utilitarian inclinations, prioritizing ethical integrity over welfare maximization. Consequently, moral obligations retain binding normative authority regardless of situational outcomes.",
    "Utilitarianism provides a consequentialist ethical framework that evaluates actions based on their contribution to overall utility and well-being. Jeremy Bentham established act utilitarianism, proposing a quantitative felicific calculus to measure aggregate pleasure and pain resulting from actions. John Stuart Mill refined this doctrine by distinguishing higher intellectual pleasures from lower sensory gratifications, arguing for qualitative moral distinctions. Rule utilitarianism contends that morality requires conformity to behavioral rules that maximize general welfare when consistently observed. Critics challenge utilitarianism for potentially overriding individual rights whenever collective happiness appears mathematically maximized.",
    "Virtue ethics conceptualizes moral character and acquired practical dispositions as the primary foundations of ethical action. Aristotle proposed in the Nicomachean Ethics that human flourishing represents the ultimate teleological objective of rational human life. Moral virtue constitutes a calibrated mean between character deficiencies and excessive behavioral extremes, requiring seasoned practical wisdom. Habitual practice transforms deliberate moral deliberation into an enduring ethical character across interpersonal circumstances. Contemporary virtue ethics emphasizes that human goodness derives from holistic personal excellence rather than rigid adherence to legalistic codes.",
    "Metaphysics investigates the fundamental nature of reality, existence, and the underlying ontological structure of the universe. Ontology categorizes entities into fundamental existential groupings, distinguishing universal properties from concrete particular objects. The problem of universals questions whether shared properties possess independent objective reality or exist merely as linguistic nominal designations. Substratum theories argue that individual particulars possess an underlying substance that supports their qualitative attributes. Modern analytical metaphysics interrogates whether modal claims about possibility and necessity reflect genuine alternative possible worlds.",
    "The mind-body problem interrogates the metaphysical relationship connecting subjective conscious experience with physical neurobiological processes. Cartesian dualism asserted that immaterial cognitive minds and extended physical bodies constitute distinct metaphysical substances that interact through the brain. Physicalism contends that mental states are strictly identical to, or supervene directly upon, neurochemical configurations within the physical central nervous system. The hard problem of consciousness questions how electrochemical signaling across neural networks gives rise to qualitative phenomenological awareness. Functionalism suggests that mental states are defined by their functional and computational roles rather than their underlying physical substrates.",
    "Phenomenology examines the structures of conscious experience systematically from the first-person perspective without presupposing external empirical claims. Edmund Husserl developed the phenomenological reduction to suspend natural assumptions about objective reality, isolating intentional conscious phenomena. Intentionality denotes the core characteristic of consciousness as always being directed toward an intended object or meaning. Martin Heidegger shifted phenomenology toward existential ontology, conceptualizing human existence as being-in-the-world immersed within historical contexts. Maurice Merleau-Ponty expanded this inquiry by demonstrating that bodily embodiment anchors conscious perception and environmental orientation.",
    "Existentialism confronts the condition of human freedom, personal responsibility, and the search for authentic meaning within an indifferent universe. Jean-Paul Sartre asserted that existence precedes essence, implying that human beings possess no predefined nature and must construct identity through choice. Authenticity requires accepting the inherent anxiety of radical freedom rather than fleeing into bad faith and societal conformity. Albert Camus explored the concept of the absurd, defining it as the irreconcilable tension between human desire for meaning and a silent universe. Embracing this absurdity allows individuals to live with defiant freedom, lucid consciousness, and creative passion.",
    "The philosophy of language investigates how linguistic expressions refer to external reality, convey semantic meaning, and facilitate communicative pragmatics. Gottlob Frege distinguished between the reference of a term and its cognitive sense, resolving puzzles regarding identity statements. Bertrand Russell developed the theory of descriptions, demonstrating that grammatical surface structure often obscures the underlying logical form of propositions. Ludwig Wittgenstein argued in his later work that linguistic meaning emerges from the practical usage of words within shared social language games. Speech act theory emphasizes that language functions performative actions, such as promising or declaring, rather than merely stating facts.",
    "Political philosophy evaluates the normative legitimacy of state governance, social contracts, and distributive justice across human societies. Thomas Hobbes argued that the state of nature represents an intolerable war of all against all, necessitating an absolute sovereign to preserve peace. John Locke posited that government legitimacy derives from the consent of the governed to protect inalienable rights to life, liberty, and property. Jean-Jacques Rousseau introduced the general will, arguing that legitimate political sovereignty must embody the common interest of the citizen collective. John Rawls formulated a modern theory of justice using the veil of ignorance, asserting that rational agents would choose principles ensuring maximum equal liberty and fair opportunity for the disadvantaged.",
]


def get_expert_paragraphs() -> List[str]:
    """Returns the 50 curated, grammatically pure domain-expert paragraphs."""
    return EXPERT_PARAGRAPHS


class ExpertCorpusDataset(Dataset):
    """
    PyTorch Dataset for causal language modeling using the 50 expert paragraphs.
    Chunks text into tokens: input_ids = tokens[:-1], labels = tokens[1:].
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 128,
        stride: int = 64,
        repeat: int = 1,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.samples = []

        all_text = "\n\n".join(EXPERT_PARAGRAPHS)
        tokens = tokenizer.encode(all_text, add_special_tokens=False)

        # Slice into chunks of length seq_len + 1 (for input and target offset)
        chunk_size = seq_len + 1
        for start_idx in range(0, len(tokens) - chunk_size + 1, stride):
            chunk = tokens[start_idx : start_idx + chunk_size]
            self.samples.append(chunk)

        # Allow dataset repetition to train for small epochs without re-tokenizing
        if repeat > 1:
            self.samples = self.samples * repeat

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class OpenWebTextStreamDataset(IterableDataset):
    """
    Streaming dataset for Hugging Face Bingsu/openwebtext_20p.
    Yields causal chunks (x, y) on-the-fly with zero RAM spikes.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 128,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.max_samples = max_samples

    def __len__(self) -> int:
        return self.max_samples if self.max_samples is not None else 5000

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        from datasets import load_dataset

        try:
            ds = load_dataset("Bingsu/openwebtext_20p", split=self.split, streaming=True)
        except Exception as e:
            # Fallback to local expert paragraphs if HuggingFace network connection drops
            print(f"⚠️ [Dataset Warning] Failed to stream OpenWebText ({e}). Falling back to expert corpus.")
            from dataset import EXPERT_PARAGRAPHS
            ds = [{"text": p} for p in EXPERT_PARAGRAPHS * 10]

        token_buffer: List[int] = []
        chunk_size = self.seq_len + 1
        samples_yielded = 0

        try:
            for row in ds:
                text = row.get("text", "")
                if not text or len(text) < 100:
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(tokens)

                while len(token_buffer) >= chunk_size:
                    chunk = token_buffer[:chunk_size]
                    token_buffer = token_buffer[self.seq_len :]

                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
                    samples_yielded += 1

                    if self.max_samples and samples_yielded >= self.max_samples:
                        return
        except Exception as e:
            print(f"⚠️ [Stream Interrupted] {e}. Yielding buffered tokens.")
            while len(token_buffer) >= chunk_size:
                chunk = token_buffer[:chunk_size]
                token_buffer = token_buffer[self.seq_len :]
                yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


class HybridCorpusDataset(Dataset):
    """
    High-performance, pre-tokenized dataset for the 40/30/30 hybrid corpus
    (Educational, Conversational, Daily Greetings & Elementary Facts).
    Pre-chunks tokens into contiguous tensors in memory so CPU data-loading
    never blocks the training loop.
    """

    def __init__(
        self,
        tokenizer,
        chunks_dir_or_file: str = "TEST/training/chunks",
        seq_len: int = 128,
        stride: int = 64,
        repeat: int = 1,
    ):
        import json
        from pathlib import Path

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        path = Path(chunks_dir_or_file)
        texts: List[str] = []

        if path.is_file():
            if path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            texts.append(json.loads(line).get("text", ""))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
        elif path.is_dir():
            # First check for jsonl chunk files
            jsonl_files = sorted(list(path.glob("train_chunk_*.jsonl")))
            if jsonl_files:
                for jf in jsonl_files:
                    with open(jf, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                texts.append(json.loads(line).get("text", ""))
            else:
                txt_file = path / "hybrid_corpus_consolidated.txt"
                if txt_file.exists():
                    with open(txt_file, "r", encoding="utf-8") as f:
                        texts.append(f.read())

        if not texts:
            # Fallback to expert paragraphs if chunks directory doesn't exist
            print(f"⚠️ [Dataset Warning] No hybrid files found in {path}. Using expert corpus.")
            texts = [p for p in EXPERT_PARAGRAPHS]

        full_text = "\n\n".join(texts)
        tokens = tokenizer.encode(full_text, add_special_tokens=False)

        chunk_size = seq_len + 1
        for start_idx in range(0, len(tokens) - chunk_size + 1, stride):
            chunk = tokens[start_idx : start_idx + chunk_size]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.samples.append((x, y))

        if repeat > 1:
            self.samples = self.samples * repeat

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


if __name__ == "__main__":
    paragraphs = get_expert_paragraphs()
    print(f"Total expert paragraphs: {len(paragraphs)}")
    for i, p in enumerate(paragraphs[:3], 1):
        sentences = [s.strip() for s in p.split(".") if s.strip()]
        print(f"\n[Paragraph {i}] ({len(sentences)} sentences):")
        print(f"  {p[:140]}...")
