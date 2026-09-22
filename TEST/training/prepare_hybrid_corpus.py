"""
================================================================================
RKMJ-Core Balanced Hybrid Corpus Preprocessor
================================================================================
Location: TEST/training/prepare_hybrid_corpus.py
Constructs a balanced, high-signal pretraining & fine-tuning dataset:
- 40% Educational / Factual Text (FineWeb-Edu / Science & Tech)
- 30% Conversational / Instruction Pairs (UltraChat / Alpaca style)
- 30% Common Daily Greetings & Elementary Encyclopedic Facts
Formats with <|im_start|> / <|im_end|> delimiters and packages into chunks.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from transformers import AutoTokenizer

# Standard Special Token Delimiters
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ==============================================================================
# 1. Bucket 3: Curated Everyday Greetings & Foundational Encyclopedic Facts (30%)
# ==============================================================================
COMMON_GREETINGS_AND_BASICS = [
    # Common conversational greetings & responses
    f"{IM_START}user\nHello!{IM_END}\n{IM_START}assistant\nHello! How can I assist you today?{IM_END}",
    f"{IM_START}user\nHi{IM_END}\n{IM_START}assistant\nHi there! What can I help you with today?{IM_END}",
    f"{IM_START}user\nhii{IM_END}\n{IM_START}assistant\nHello! How are you doing today? Let me know how I can help you.{IM_END}",
    f"{IM_START}user\nGood morning!{IM_END}\n{IM_START}assistant\nGood morning! I hope you have a productive day. How can I help you?{IM_END}",
    f"{IM_START}user\nGood evening!{IM_END}\n{IM_START}assistant\nGood evening! How can I be of assistance tonight?{IM_END}",
    f"{IM_START}user\nWho are you?{IM_END}\n{IM_START}assistant\nI am an intelligent, lightweight AI assistant powered by the RKMJ 1.58-bit neural framework.{IM_END}",
    f"{IM_START}user\nWhat is your name?{IM_END}\n{IM_START}assistant\nI am RKMJ Assistant, a high-efficiency 1.58-bit language model.{IM_END}",
    f"{IM_START}user\nHow are you?{IM_END}\n{IM_START}assistant\nI am functioning smoothly and ready to assist you. What topic would you like to explore?{IM_END}",
    f"{IM_START}user\nThank you!{IM_END}\n{IM_START}assistant\nYou are very welcome! Feel free to ask if you have any more questions.{IM_END}",
    f"{IM_START}user\nThanks for your help.{IM_END}\n{IM_START}assistant\nIt was my pleasure to help. Have a wonderful day!{IM_END}",
    f"{IM_START}user\nBye{IM_END}\n{IM_START}assistant\nGoodbye! Have a great day ahead.{IM_END}",

    # Elementary facts & common definitions
    f"{IM_START}user\nWhat is the capital of France?{IM_END}\n{IM_START}assistant\nThe capital of France is Paris. Paris is also the country's most populous city and its cultural and economic center.{IM_END}",
    f"{IM_START}user\nWhat is water made of?{IM_END}\n{IM_START}assistant\nWater is composed of hydrogen and oxygen. Its chemical formula is H2O, meaning each molecule contains two hydrogen atoms bonded to one oxygen atom.{IM_END}",
    f"{IM_START}user\nWhat is the largest planet in our solar system?{IM_END}\n{IM_START}assistant\nJupiter is the largest planet in our solar system. It is a gas giant primarily composed of hydrogen and helium.{IM_END}",
    f"{IM_START}user\nHow many days are in a year?{IM_END}\n{IM_START}assistant\nA standard calendar year has 365 days. A leap year occurs every four years and contains 366 days.{IM_END}",
    f"{IM_START}user\nWhat is the speed of light?{IM_END}\n{IM_START}assistant\nThe speed of light in a vacuum is approximately 299,792,458 meters per second, commonly rounded to 300,000 kilometers per second.{IM_END}",
    f"{IM_START}user\nWhat is gravity?{IM_END}\n{IM_START}assistant\nGravity is a natural phenomenon by which all physical bodies with mass or energy are drawn toward one another.{IM_END}",
    f"{IM_START}user\nWhat causes day and night on Earth?{IM_END}\n{IM_START}assistant\nDay and night are caused by the rotation of the Earth on its axis. As Earth rotates, different parts face toward or away from the Sun.{IM_END}",
    f"{IM_START}user\nWhat is an atom?{IM_END}\n{IM_START}assistant\nAn atom is the basic building block of all matter. It consists of a dense central nucleus of protons and neutrons surrounded by a cloud of negatively charged electrons.{IM_END}",
    f"{IM_START}user\nWhat is the boiling point of water?{IM_END}\n{IM_START}assistant\nAt standard sea-level atmospheric pressure, water boils at 100 degrees Celsius or 212 degrees Fahrenheit.{IM_END}",
    f"{IM_START}user\nWhy is the sky blue?{IM_END}\n{IM_START}assistant\nThe sky appears blue because molecules in Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other wavelengths due to Rayleigh scattering.{IM_END}",
    f"{IM_START}user\nWhat is 2 + 2?{IM_END}\n{IM_START}assistant\n2 + 2 equals 4.{IM_END}",
    f"{IM_START}user\nWhat is the capital of India?{IM_END}\n{IM_START}assistant\nThe capital of India is New Delhi. It serves as the seat of the executive, legislative, and judicial branches of the government of India.{IM_END}",
    f"{IM_START}user\nWhat is photosynthesis?{IM_END}\n{IM_START}assistant\nPhotosynthesis is the biological process used by plants and algae to convert light energy into chemical energy stored in glucose.{IM_END}",
    f"{IM_START}user\nCan you tell me a fun fact?{IM_END}\n{IM_START}assistant\nHoney never spoils. Archaeologists have discovered pots of honey in ancient Egyptian tombs that are over three thousand years old and still completely edible.{IM_END}",
]

# ==============================================================================
# 2. Bucket 2: Conversational / Instructional Pairs (30%)
# ==============================================================================
CONVERSATIONAL_INSTRUCTION_SAMPLES = [
    f"{IM_START}user\nExplain the concept of supply and demand in simple terms.{IM_END}\n{IM_START}assistant\nSupply refers to the total amount of a specific good or service available to consumers. Demand represents how much consumers desire that product and are willing to pay for it. When supply exceeds demand, prices typically fall. Conversely, when demand outstrips supply, prices tend to rise until market equilibrium is reached.{IM_END}",
    f"{IM_START}user\nWrite a short summary of how the internet works.{IM_END}\n{IM_START}assistant\nThe internet is a global network of interconnected computers that communicate using standardized communication protocols such as TCP/IP. Data is divided into small units called packets and routed across optical fiber cables, satellites, and routers to reach its destination. Domain Name System servers translate human-readable domain names into numerical IP addresses to locate servers.{IM_END}",
    f"{IM_START}user\nWhat are the primary differences between Python and C++?{IM_END}\n{IM_START}assistant\nPython is an interpreted, dynamically typed language known for clean syntax and rapid prototyping. In contrast, C++ is a compiled, statically typed language that provides manual memory management and low-level hardware control, achieving near-optimal execution speed.{IM_END}",
    f"{IM_START}user\nHow can I improve my focus when studying?{IM_END}\n{IM_START}assistant\nTo improve focus, eliminate digital distractions by silencing notifications. Use structured techniques such as the Pomodoro technique, working in uninterrupted 25-minute intervals followed by brief 5-minute breaks. Maintain a dedicated, uncluttered workspace and stay well-hydrated.{IM_END}",
    f"{IM_START}user\nWhat is the purpose of an operating system?{IM_END}\n{IM_START}assistant\nAn operating system manages computer hardware and software resources. It provides essential services such as process scheduling, memory allocation, file system management, and device drivers, serving as an intermediary between user applications and the physical CPU, RAM, and storage.{IM_END}",
    f"{IM_START}user\nCan you explain what an algorithm is?{IM_END}\n{IM_START}assistant\nAn algorithm is a finite, unambiguous sequence of instructions designed to solve a specific problem or perform a computation. Each step must be clearly defined and lead toward a determinable conclusion.{IM_END}",
    f"{IM_START}user\nWhat is machine learning?{IM_END}\n{IM_START}assistant\nMachine learning is a subfield of artificial intelligence where computer systems learn patterns and decision rules directly from data rather than following explicitly programmed static instructions. Models improve their performance on specific tasks through statistical training algorithms.{IM_END}",
    f"{IM_START}user\nWhat is the scientific method?{IM_END}\n{IM_START}assistant\nThe scientific method is an empirical procedure for acquiring knowledge. It involves making systematic observations, formulating testable hypotheses, conducting controlled experiments, analyzing data, and drawing evidence-based conclusions that can be independently replicated.{IM_END}",
    f"{IM_START}user\nHow do vaccines protect the human body?{IM_END}\n{IM_START}assistant\nVaccines introduce an inactivated or harmless component of a pathogen into the body. This trains the adaptive immune system to recognize specific antigens and produce antibodies and memory cells, providing rapid defense if exposed to the actual pathogen in the future.{IM_END}",
    f"{IM_START}user\nWhat is renewable energy?{IM_END}\n{IM_START}assistant\nRenewable energy is energy derived from natural resources that replenish themselves continuously on a human timescale, such as solar power, wind energy, hydroelectric power, and geothermal energy.{IM_END}",
]

# ==============================================================================
# 3. Bucket 1: Educational & Factual Knowledge Passages (40%)
# ==============================================================================
EDUCATIONAL_FACTUAL_FALLBACKS = [
    "Photosynthesis converts electromagnetic solar radiation into stable chemical potential energy within organic carbohydrates. The light-dependent reactions occur across the thylakoid membranes of chloroplasts, where photon absorption initiates the photolysis of water molecules. This catalytic dissociation produces gaseous oxygen, protons, and high-energy electrons that traverse an organized electron transport chain. The resulting electrochemical proton gradient drives adenosine triphosphate synthase to produce ATP and NADPH. These energetic intermediates fuel the Calvin cycle within the stroma, wherein carbon dioxide undergoes enzymatic fixation by Rubisco to synthesize three-carbon phosphoglycerate precursors.",
    "Cellular respiration oxidizes hexose sugars through a sequence of convergent metabolic pathways to generate universal biochemical energy. Glycolysis initiates the enzymatic cleavage of glucose into two pyruvate molecules within the aqueous cytoplasm, yielding a net gain of two ATP molecules and two NADH molecules. Under aerobic conditions, pyruvate translocates across the mitochondrial membrane, where the pyruvate dehydrogenase complex decarboxylates it into acetyl coenzyme A. Acetyl groups enter the citric acid cycle, undergoing cyclical oxidation to release carbon dioxide while reducing electron carriers. The resulting high-energy electrons drive oxidative phosphorylation, establishing the transmembrane proton motive force that synthesizes cellular ATP.",
    "Linear algebra analyzes the geometric and algebraic properties of vector spaces, linear transformations, and finite-dimensional matrices. A linear transformation maps vectors between vector spaces while rigorously preserving vector addition and scalar multiplication operations. The spectral theorem guarantees that every real symmetric matrix possesses an orthogonal basis of eigenvectors associated with purely real eigenvalues. Matrix diagonalization decomposes a linear operator into canonical diagonal form, revealing invariant dimensional directions. Singular value decomposition generalizes this factorization to arbitrary rectangular matrices, providing computational foundations for numerical optimization and dimensionality reduction.",
    "Calculus formalizes the continuous variation and geometric accumulation of quantities through differential and integral operations. The derivative quantifies the instantaneous rate of change of a differentiable function as the limiting ratio of functional increments to input increments. The Riemann integral aggregates infinitesimal area partitions beneath a curve, formalizing continuous summation across bounded domains. The fundamental theorem of calculus establishes that differentiation and definite integration serve as mutually inverse operations. Taylor series expansions further approximate sufficiently smooth analytic functions as infinite polynomial series centered around a localized point.",
    "Plate tectonics governs the large-scale kinematic displacement and dynamic recycling of the lithosphere across the convective asthenosphere. Convection currents within the silicate mantle drive divergent boundaries where seafloor spreading creates new basaltic oceanic crust along mid-ocean ridge systems. Convergent plate margins produce subduction zones, forcing denser oceanic lithosphere beneath continental margins and generating deep oceanic trenches alongside volcanic island arcs. Transform faults accommodate horizontal displacement between adjacent tectonic plates, periodically releasing accumulated elastic strain as seismic earthquake ruptures.",
    "The Neolithic Revolution transitioned human societies from nomadic hunting and gathering to sedentary agricultural civilization. The deliberate domestication of cereal grains such as emmer wheat and barley fostered permanent settlements across the fertile crescent of southwest Asia. Increased caloric yields generated agricultural surpluses, which catalyzed demographic population expansion and occupational specialization. Craft production, metallurgy, and centralized administrative hierarchies emerged to manage surplus distribution and territorial boundaries. This societal transformation established the fundamental prerequisite conditions for early urbanism.",
    "The Scientific Revolution transformed epistemology by establishing empirical observation, controlled experimentation, and mathematical formulation as the standard criteria for scientific truth. Nicolaus Copernicus formulated a mathematically coherent heliocentric astronomical model, challenging geocentric Ptolemaic dogma. Galileo Galilei utilized optical telescopes to observe lunar topography, Jovian moons, and planetary phases, providing observational corroboration for heliocentrism. Johannes Kepler derived mathematical laws of planetary motion, demonstrating that celestial orbits form ellipses rather than uniform circular paths. Isaac Newton integrated these discoveries into the universal law of gravitation.",
    "Computability theory establishes the theoretical mathematical limits of algorithmic problem solving through formal models of computation. Alan Turing formulated the Turing machine, demonstrating that an abstract state machine manipulating a linear tape accurately captures mechanical computation. The Church-Turing thesis conjectures that any physically realizable algorithmic process can be simulated by a universal Turing machine. The halting problem establishes that no general algorithm can decide whether an arbitrary program will eventually terminate or execute infinitely.",
    "Information theory formalizes the quantification, storage, and reliable communication of discrete data across noisy communication channels. Claude Shannon defined information entropy as the expected value of the self-information contained within stochastic message outcomes. Shannon noiseless source coding theorem establishes the fundamental limit for lossless data compression as the entropy rate of the source. The noisy-channel coding theorem guarantees that information can be transmitted with arbitrarily low error rates provided the transmission rate remains below channel capacity.",
    "Epistemology investigates the fundamental nature, acquisition, and normative justification of human knowledge and rational belief. Classical analysis defines knowledge as justified true belief, demanding that an agent hold an epistemically warrantable true proposition. The Gettier problem challenged this tripartite definition by presenting scenarios where justified true beliefs arise through epistemic luck. Foundationalism posits that knowledge rests upon self-justifying basic beliefs that require no further inferential support.",
]


import time
from tqdm import tqdm


def fetch_fineweb_edu_samples(count: int) -> List[str]:
    """Streams educational text samples from HuggingFaceFW/fineweb-edu with resilient retry & backoff."""
    samples: List[str] = []
    max_retries = 8
    retry_delay = 2.0
    pbar = tqdm(total=count, desc="Fetching FineWeb-Edu (40%)", unit="samples")

    for attempt in range(max_retries):
        if len(samples) >= count:
            break
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            for row in ds:
                text = row.get("text", "").strip()
                if 150 <= len(text) <= 2500:
                    samples.append(text)
                    pbar.update(1)
                    if len(samples) >= count:
                        break
        except Exception as e:
            if attempt < max_retries - 1 and len(samples) < count:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30.0)
            else:
                break

    pbar.close()

    if len(samples) < count:
        needed = count - len(samples)
        print(f"ℹ️ [FineWeb] Supplemented {needed} educational samples with curated academic corpus.")
        multiplier = (needed // len(EDUCATIONAL_FACTUAL_FALLBACKS)) + 1
        samples.extend((EDUCATIONAL_FACTUAL_FALLBACKS * multiplier)[:needed])

    return samples[:count]


def fetch_instruction_samples(count: int) -> List[str]:
    """Streams conversational/instruction pairs from UltraChat with resilient retry & backoff."""
    samples: List[str] = []
    max_retries = 8
    retry_delay = 2.0
    pbar = tqdm(total=count, desc="Fetching UltraChat (30%)", unit="pairs")

    for attempt in range(max_retries):
        if len(samples) >= count:
            break
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
            for row in ds:
                messages = row.get("messages", [])
                if len(messages) >= 2:
                    u_msg = messages[0].get("content", "").strip()
                    a_msg = messages[1].get("content", "").strip()
                    if u_msg and a_msg and len(u_msg) < 400 and len(a_msg) < 1500:
                        formatted = f"{IM_START}user\n{u_msg}{IM_END}\n{IM_START}assistant\n{a_msg}{IM_END}"
                        samples.append(formatted)
                        pbar.update(1)
                        if len(samples) >= count:
                            break
        except Exception as e:
            if attempt < max_retries - 1 and len(samples) < count:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30.0)
            else:
                break

    pbar.close()

    if len(samples) < count:
        needed = count - len(samples)
        print(f"ℹ️ [Instruction] Supplemented {needed} conversational samples with curated dialogue corpus.")
        multiplier = (needed // len(CONVERSATIONAL_INSTRUCTION_SAMPLES)) + 1
        samples.extend((CONVERSATIONAL_INSTRUCTION_SAMPLES * multiplier)[:needed])

    return samples[:count]


def build_hybrid_corpus(
    output_dir: str = "TEST/training/chunks",
    num_samples: int = 600,
    chunk_tokens: int = 128,
    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    chunks_per_file: int = 1000,
    batch_write_size: int = 1000,
) -> List[str]:
    """
    Builds the 40/30/30 balanced dataset, tokenizes with buffered streaming,
    and writes standardized chunk files without unbounded memory growth.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📦 RKMJ Large-Scale Hybrid Corpus Builder")
    print(f"   Target Samples:     {num_samples:,}")
    print(f"   Tokens per Chunk:   {chunk_tokens}")
    print(f"   Chunks per File:    {chunks_per_file:,}")
    print(f"   Ratio:              40% Educational | 30% Conversational | 30% Common Basics")
    print(f"   Tokenizer:          {tokenizer_name}")
    print("=" * 80)

    # 1. Calculate sample counts according to target ratio
    n_edu = int(num_samples * 0.40)
    n_conv = int(num_samples * 0.30)
    n_basics = num_samples - n_edu - n_conv

    print(f"\n[*] Step 1/3: Gathering dataset mix ({n_edu:,} Edu, {n_conv:,} Conv, {n_basics:,} Basics)...")

    # Bucket 1: Educational (40%)
    edu_data = fetch_fineweb_edu_samples(n_edu)

    # Bucket 2: Conversational (30%)
    conv_data = fetch_instruction_samples(n_conv)

    # Bucket 3: Basic Greetings & Everyday Facts (30%)
    multiplier = (n_basics // len(COMMON_GREETINGS_AND_BASICS)) + 1
    basics_data = (COMMON_GREETINGS_AND_BASICS * multiplier)[:n_basics]

    # Combine & Shuffle
    all_raw_samples = edu_data + conv_data + basics_data
    random.seed(42)
    random.shuffle(all_raw_samples)
    print(f"✅ Collected & shuffled {len(all_raw_samples):,} balanced entries.")

    # 2. Tokenize and write with buffered streaming
    print(f"\n[*] Step 2/3: Tokenizing and buffered chunk writing...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    consolidated_file = out_path / "hybrid_corpus_consolidated.txt"
    # Reset consolidated file
    with open(consolidated_file, "w", encoding="utf-8") as f:
        pass

    chunk_files: List[str] = []
    chunk_idx = 1
    current_chunk_tokens: List[int] = []
    file_chunks: List[str] = []
    total_chunks_written = 0

    pbar_write = tqdm(total=len(all_raw_samples), desc="Processing & Writing Chunks", unit="entries")

    for i in range(0, len(all_raw_samples), batch_write_size):
        batch = all_raw_samples[i : i + batch_write_size]

        # Append raw text to consolidated file
        with open(consolidated_file, "a", encoding="utf-8") as cf:
            for entry in batch:
                cf.write(entry.strip() + "\n\n")

        # Tokenize batch
        for entry in batch:
            tokens = tokenizer.encode(entry, add_special_tokens=False)
            current_chunk_tokens.extend(tokens)

            while len(current_chunk_tokens) >= chunk_tokens:
                chunk = current_chunk_tokens[:chunk_tokens]
                current_chunk_tokens = current_chunk_tokens[chunk_tokens:]
                decoded_chunk = tokenizer.decode(chunk, clean_up_tokenization_spaces=False)
                file_chunks.append(decoded_chunk)

                if len(file_chunks) >= chunks_per_file:
                    chunk_file = out_path / f"train_chunk_{chunk_idx:04d}.jsonl"
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        for c in file_chunks:
                            f.write(json.dumps({"text": c}) + "\n")
                    chunk_files.append(str(chunk_file))
                    total_chunks_written += len(file_chunks)
                    chunk_idx += 1
                    file_chunks = []

        pbar_write.update(len(batch))

    pbar_write.close()

    # Flush any remaining chunks
    if file_chunks:
        chunk_file = out_path / f"train_chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for c in file_chunks:
                f.write(json.dumps({"text": c}) + "\n")
        chunk_files.append(str(chunk_file))
        total_chunks_written += len(file_chunks)

    print("\n" + "=" * 80)
    print("✅ Hybrid Corpus Build Complete!")
    print(f"   Total Chunks Created:  {total_chunks_written:,} (each {chunk_tokens} tokens)")
    print(f"   Chunk Files Written:   {len(chunk_files)} (.jsonl)")
    print(f"   Consolidated File:     {consolidated_file}")
    print("=" * 80)
    return chunk_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Balanced Hybrid Training Corpus")
    parser.add_argument("--output-dir", type=str, default="TEST/training/chunks")
    parser.add_argument("--num-samples", type=int, default=1000, help="Total samples (supports up to 50,000+)")
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--chunks-per-file", type=int, default=1000)
    parser.add_argument("--tokenizer", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    args = parser.parse_args()

    build_hybrid_corpus(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        chunk_tokens=args.chunk_tokens,
        tokenizer_name=args.tokenizer,
        chunks_per_file=args.chunks_per_file,
    )
