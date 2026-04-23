# Bangla TTS Pipeline: Architecture & Workflows

This document centralizes the visual workflows of the Bangla TTS pipeline, from raw input to final audio synthesis.

````carousel
### **1. High-Level Pipeline Architecture**
Objective: End-to-end flow from raw text to waveform.

```mermaid
graph TD
    Input[Raw Text Input] --> Unicode[Unicode Normalization /NFC/]
    
    subgraph Normalization["Hybrid Normalization Engine"]
        Unicode --> DE[Decision Engine]
        DE --> Classify{Classifier}
        Classify -- UNIT --> UH[Unit Handler]
        Classify -- ABBREV --> AH[Abbreviation Handler]
        Classify -- MIXED --> TS[Token Splitter]
        TS --> DE
        
        UH --> RuleStage[Rule-Based Expansion]
        AH --> RuleStage
        
        RuleStage --> Rules["Numbers, Dates, Currency, Time"]
        Rules --> NER[NER Handler]
    end

    subgraph Context["Contextual Refinement"]
        NER --> ML{Use ML?}
        ML -- Yes --> T5[ML Translator /BanglaT5/]
        T5 --> PostCleanup[Post-ML Cleanup]
        ML -- No --> G2P
        PostCleanup --> G2P
    end

    subgraph G2P["Hybrid G2P Engine"]
        G2P[Grapheme-to-Phoneme] --> L1[Layer 1: Dictionary Lookup]
        L1 -- Miss --> L2[Layer 2: Rule-Based Mapping]
        L2 -- Miss --> L3[Layer 3: Neural G2P /byt5/]
        L3 -- Fallback --> FinishG2P[Phoneme Stream]
    end

    subgraph TTS["Audio Synthesis"]
        FinishG2P --> Prosody[Prosody Formatter]
        Prosody --> Engine{TTS Engine}
        Engine -- GTTS --> Cloud[Google Cloud TTS]
        Engine -- VITS2 --> Neural[VITS2 Model]
        Cloud --> Output[Audio Output]
        Neural --> Output
    end
```

<!-- slide -->
### **2. Step 1: Token Classification**
Objective: Route tokens to specialized handlers.

```mermaid
graph LR
    Token[Input Token] --> Regex{Regex Pattern?}
    Regex -- "Digit+String" --> Mixed[MIXED]
    Regex -- "JSON Keys match" --> Abbrev[ABBREVIATION]
    Regex -- "Unit list match" --> Unit[UNIT]
    Regex -- "None of above" --> Normal[NORMAL]
```

<!-- slide -->
### **3. Step 2 & 3: Specialized Handlers & Semantic Expansion**
Objective: Expand tokens into gramatically correct Bengali.

```mermaid
graph TD
    subgraph Specialist["Specialist Processing"]
        Unit[UNIT] --> UH[UnitHandler: 5kg -> ৫ কেজি]
        Abbrev[ABBREVIATION] --> AH[AbbrevHandler: Dr. -> ডাক্তার]
        Mixed[MIXED] --> Split[TokenSplitter: 10GB -> 10 + GB]
        Split --> Recurse[Recurse Decision Engine]
    end

    subgraph Expansion["Semantic Expansion"]
        UH & AH & Recurse --> Date[Month/Date Rules]
        Date --> Currency[Currency Rules]
        Currency --> Numbers[Number Rules]
        Numbers --> NER[NER Handler]
    end
```

<!-- slide -->
### **4. Step 4: Hybrid G2P Engine**
Objective: Convert text to phonetic representations.

```mermaid
graph TD
    Word[Bangla Word] --> L1[Layer 1: Dict Lookup]
    L1 -- Miss --> L2[Layer 2: Rule Mapping]
    L2 -- Miss --> L3[Layer 3: Neural Model /byt5/]
    L3 -- Fail --> L4[Layer 4: Fallback Original]
    
    subgraph Rules["Rule Logic"]
        L2 --> Cons[Consonant Map]
        L2 --> Vowel[Vowel/Matra Map]
        L2 --> Hasanta[Hasanta Suppression]
    end
```

<!-- slide -->
### **5. Step 5: Synthesis & Prosody**
Objective: Final formatting and wave generation.

```mermaid
graph LR
    Phonemes[Phoneme Stream] --> Prosody[Prosody Formatter]
    Prosody --> Mode{Engine Selection}
    Mode -- "Cloud" --> GTTS[GTTS Backend]
    Mode -- "Local" --> VITS2[VITS2 Backend]
    GTTS & VITS2 --> Wave[Waveform Generation]
```
````
