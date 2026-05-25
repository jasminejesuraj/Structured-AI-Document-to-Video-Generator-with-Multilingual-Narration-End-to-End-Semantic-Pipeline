import sys
try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import re
    import fitz  
    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.schema import Document
    import requests
    import json
    import math
    from typing import List, Dict
    import sys
    import subprocess
    import time

    def ensure_ollama_running():

        try:
            requests.get("http://localhost:11434")
            print("Ollama already running")
            return

        except:
            print("Starting Ollama")

            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            time.sleep(5)

            print("Ollama started")
    def ensure_model_exists(model_name="llama3.2:3b"):

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False
                },
                timeout=10
            )

            if response.status_code == 200:
                print(f"{model_name} already exists")
                return

        except:
            pass

        print(f"Pulling {model_name}...")

        subprocess.run(["ollama", "pull", model_name])

        print("Model pulled")
        
    MIN_WORDS_PER_SECTION = 120     
    SEMANTIC_SPLIT_MIN_WORDS = 300  

    SCENE_TYPES = ["Intro", "Explanation", "Example", "Conclusion"]

    # STEP 1: SECTION ID EXTRACTION 

    def get_section_id(heading: str):
        match = re.match(r"^(\d+(?:\.\d+)*)", heading)
        return match.group(1) if match else None



    #  STEP 2: HEADING DETECTION 

    def is_heading(line: str) -> bool:

        line = line.strip()

        if not line:
            return False

        # bullets
        if line.startswith(("●", "•", "-", "–", "", "*")):
            return False

        words = line.split()

        # micro / overly long
        if len(words) <= 2 or len(words) > 12:
            return False

        if re.search(
            r"\b(accepted|received|published|available online)\b",
            line.lower()
        ):
            return False

        if is_equation_line(line):
            return False

        symbol_ratio = (
            len(re.findall(r"[^a-zA-Z0-9\s]", line))
            / max(1, len(line))
        )

        if symbol_ratio > 0.25:
            return False

        # numbered headings
        if re.match(r"^\d+(\.\d+)*\s+.+", line):
            return True

        if line.isupper() or line.istitle():
            return True

        return False


    # STEP 3: TABLE / SURVEY REWRITE

    def rewrite_table_like_lines(lines):
        rewritten = []

        for line in lines:
            if "✅" in line or "❌" in line:
                parts = re.split(r"\s{2,}", line)

                if len(parts) >= 3:
                    entity = parts[0].strip()
                    yes_col = parts[1]
                    no_col = parts[2]

                    if "Yes" in yes_col and "No" in no_col:
                        rewritten.append(f"{entity} benefits from anonymity.")
                    elif "No" in yes_col and "Yes" in no_col:
                        rewritten.append(f"{entity} requires verified identities.")
                    else:
                        rewritten.append(line)
                else:
                    rewritten.append(line)
            else:
                rewritten.append(line)

        return rewritten


    #STEP 4: STRUCTURE-AWARE EXTRACTION 

    def extract_structured_sections(pdf_path):
        sections = []
        current_heading = None
        current_content = []
        current_region = "BODY"

        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                page_height = page.rect.height
                raw_blocks = page.get_text("blocks")
                blocks = []
                for b in raw_blocks:
                    x0, y0, x1, y1, text, *_ = b
                    if y0 < page_height * 0.08:
                        continue                
                    if y1 > page_height * 0.92:
                        continue
                    blocks.append(b)
                page_width = page.rect.width

                if is_two_column_layout(blocks, page_width):
                    middle_x = page_width / 2
                    left_blocks = []
                    right_blocks = []
                    for b in blocks:
                        x0, y0, x1, y1, text, *_ = b
                        if x0 < middle_x:
                            left_blocks.append(b)
                        else:
                            right_blocks.append(b)

                    left_blocks.sort(key=lambda b: b[1])
                    right_blocks.sort(key=lambda b: b[1])
                    blocks = left_blocks + right_blocks
                else:
                    blocks.sort(key=lambda b: (b[1], b[0]))
                lines = []
                for b in blocks:
                    lines.extend(b[4].splitlines())
                lines = fix_broken_lines(lines)

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    special = detect_special_section(line)
                    if is_noise_line(line):
                        continue
                    if not line:
                        continue
                    if special:
                        current_region = special
                        continue
                    if current_region in ["REFERENCES", "ACKNOWLEDGEMENT"]:
                        continue
                    if is_caption(line):
                        continue
                    if is_equation_line(line):
                        continue

                    if is_heading(line):
                        if current_heading and current_content:
                            cleaned = clean_heading(current_heading)

                            sections.append(
                                Document(
                                    text=cleaned + "\n" + "\n".join(current_content),
                                    metadata={"heading": cleaned}
                                )
                            )

                        current_heading = line
                        current_content = []
                    else:
                        if current_heading:
                            cleaned = rewrite_table_like_lines([line])
                            current_content.extend(cleaned)

        if current_heading and current_content:
            cleaned = clean_heading(current_heading)

            sections.append(
                Document(
                    text=cleaned + "\n" + "\n".join(current_content),
                    metadata={"heading": cleaned}
                )
            )

        return sections


    # STEP 5: MERGE SMALL SECTIONS 

    def merge_small_sections(sections, min_words=MIN_WORDS_PER_SECTION):
        merged = []

        buffer_text = ""
        buffer_heading = None
        buffer_section_id = None

        for doc in sections:
            heading = doc.metadata.get("heading")
            section_id = get_section_id(heading)
            word_count = len(doc.text.split())

            if word_count < min_words:
                if not buffer_text:
                    buffer_text = doc.text
                    buffer_heading = heading
                    buffer_section_id = section_id
                else:
                    
                    if section_id == buffer_section_id:
                        buffer_text += "\n" + doc.text
                    else:
                        merged.append(
                            Document(
                                text=buffer_text,
                                metadata={"heading": buffer_heading}
                            )
                        )
                        buffer_text = doc.text
                        buffer_heading = heading
                        buffer_section_id = section_id
            else:
                if buffer_text:
                    merged.append(
                        Document(
                            text=buffer_text,
                            metadata={"heading": buffer_heading}
                        )
                    )
                    buffer_text = ""
                    buffer_heading = None
                    buffer_section_id = None

                merged.append(doc)

        if buffer_text:
            merged.append(
                Document(
                    text=buffer_text,
                    metadata={"heading": buffer_heading}
                )
            )

        return merged

    # STEP 6: SEMANTIC SPLITTER 
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95
    )

    # LLaMA UTILITIES 
    def clean_heading(heading: str) -> str:

        if not heading:
            return heading

        # patterns
        heading = re.sub(r'^\d+(\.\d+)*', '', heading)

        # punctuation and spaces 
        heading = re.sub(r'^[\s\.\-\)\:]+', '', heading)

        return heading.strip()

    def fix_broken_lines(lines):
        
        fixed = []
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            if i + 1 < len(lines):
                next_line = lines[i + 1].lstrip()

                # hyphenated 
                if line.endswith("-"):
                    fixed.append(line[:-1] + next_line)
                    i += 2
                    continue

                #  mid-word
                if (
                    line
                    and next_line
                    and line[-1].isalpha()
                    and next_line[0].islower()
                    and not line.endswith((".", ":", ";", "?", "!"))
                ):
                    fixed.append(line + next_line)
                    i += 2
                    continue

            fixed.append(line)
            i += 1

        return fixed
    def estimate_duration(narration_text, wps=2.5):
        """
        3 words per second (~180 wpm)
        """
        words = len(narration_text.split())
        base = words / wps

        return max(3, math.ceil(base + 1))

    def infer_topic(text: str) -> str:

        if "responsibil" in text.lower():
            return "Roles & Responsibilities"
        elif "skill" in text.lower():
            return "Skills Required"
        elif "client" in text.lower():
            return "Client Interaction"
        else:
            return "Overview"
        
    def generate_with_llama(prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
        "temperature": 0.3
            }
            },
            timeout=120
        )
        return response.json()["response"]
        
    def save_storyboard(storyboard: List[Dict], out_path: str = "storyboard.json"):
        with open(out_path, "w", encoding="utf-8") as f:
            #json.dump(storyboard, f, ensure_ascii=False, indent=2)
            final_json = {
                "project": {
                    "source_pdf": pdf_path,
                    "total_scenes": len(storyboard),
                    "pipeline_version": "1.0"
                },
                "scenes": storyboard
            }

            json.dump(final_json, f, ensure_ascii=False, indent=2)
        print(f"Storyboard saved to {out_path}")

    def is_noise_line(line: str) -> bool:

        patterns = [
            r"^\d+$",  
            r"doi",
            r"www\.",
            r"all rights reserved",
            r"journal",
            r"neural networks",
            r"published by",
        ]

        line_lower = line.lower()

        return any(re.search(p, line_lower) for p in patterns)

    def is_caption(line: str) -> bool:

        line = line.strip().lower()

        patterns = [
            r"^figure\s+\d+",
            r"^fig\.\s*\d+",
            r"^table\s+\d+",
        ]

        return any(re.match(p, line) for p in patterns)
    
    def is_equation_line(line: str) -> bool:

        math_symbols = r"[=+\-*/∑∫√≈≠≤≥∆∂λμσπθ]"
        symbol_count = len(re.findall(math_symbols, line))
        ratio = len(re.findall(r"[^a-zA-Z\s]", line)) / max(1, len(line))

        latex_patterns = [
            r"\\begin",
            r"\\end",
            r"\$.*?\$",
            r"\\frac",
            r"\\sum",
        ]

        if symbol_count >= 3:
            return True

        if ratio > 0.4:
            return True

        if any(re.search(p, line) for p in latex_patterns):
            return True

        return False
    def detect_special_section(line: str):

        line = line.strip().lower()

        mapping = {
            "abstract": "ABSTRACT",
            "references": "REFERENCES",
            "bibliography": "REFERENCES",
            "acknowledgement": "ACKNOWLEDGEMENT",
            "acknowledgments": "ACKNOWLEDGEMENT",
            "appendix": "APPENDIX",
        }

        for k, v in mapping.items():
            if line.startswith(k):
                return v

        return None
    def clean_narration(text: str) -> str:

        text = text.replace("->", " to ")

        text = text.replace('"', '')
        text = text.replace("“", "")
        text = text.replace("”", "")

    
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"\bClick\b", "Select", text, flags=re.IGNORECASE)

        text = re.sub(r"[<>|{}[\]]", "", text)

        text = re.sub(r"\b(OK|Cancel|Apply|Finish|Next)\b", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

        
    def sanitize_text(text: str) -> str:

        text = re.sub(r"[A-Z]:\\\\[^\s]+", "", text)          #  paths
        text = re.sub(r"/[^\s]+(?:/[^\s]+)+", "", text)      

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)


        text = re.sub(r"\S+@\S+", "", text)

        code_patterns = [
            r"`[^`]+`",                         
            r"\bdef\s+\w+\(.*?\):",            
            r"\bclass\s+\w+",                  
            r"#include\s*<.*?>",               
            r"public\s+static\s+void\s+main",   
            r"System\.out\.println",
            r"console\.log",
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            r"\w+\(.*?\);",                    
        ]

        for p in code_patterns:
            text = re.sub(p, "", text)

        terminal_patterns = [
            r"\bpip\s+install\b.*",
            r"\bpython\s+\S+",
            r"\bcd\s+\S+",
            r"\bls\b",
            r"\bdir\b",
            r"\bmkdir\b",
            r"\bgit\s+\w+",
            r"\bnpm\s+\w+",
        ]

        for p in terminal_patterns:
            text = re.sub(p, "", text)

        text = re.sub(r"->", " to ", text)
        ui_patterns = [
            r"\bclick\b",
            r"\bselect\b",
            r"\bpress\b",
            r"\bbutton\b",
            r"\bmenu\b",
            r"\btab\b",
        ]

        for p in ui_patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

        text = re.sub(r"[{}[\]|<>_=+~`]", "", text)

        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"\-{2,}", "-", text)

        lines = text.splitlines()

        cleaned_lines = []

        for line in lines:

            line = line.strip()
            if len(line) < 3:
                continue
            if re.search(r"[;{}<>]", line):
                continue
            symbol_ratio = len(re.findall(r"[^a-zA-Z0-9\s]", line)) / max(1, len(line))
            if symbol_ratio > 0.35:
                continue
            cleaned_lines.append(line)

        text = " ".join(cleaned_lines)

        text = text.replace('"', '')
        text = text.replace("“", "")
        text = text.replace("”", "")

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def is_two_column_layout(blocks, page_width):

        middle_x = page_width / 2

        left_count = 0
        right_count = 0

        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            if len(text.split()) < 5:
                continue

            if x0 < middle_x:
                left_count += 1
            else:
                right_count += 1

        total = left_count + right_count

        if total == 0:
            return False

        left_ratio = left_count / total
        right_ratio = right_count / total

        return left_ratio > 0.3 and right_ratio > 0.3
    
    # STEP 7: SPLIT LARGE CHUNKS
    def split_large_chunk(chunk: Document, max_words: int = 150) -> List[Document]:

        text = chunk.text.strip()

        # sentence-aware splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        sub_chunks = []

        current = []
        current_words = 0

        for sentence in sentences:

            sentence_words = len(sentence.split())

            if current_words + sentence_words > max_words:

                combined = " ".join(current)

                sub_chunks.append(
                    Document(
                        text=combined,
                        metadata={
                            "heading": chunk.metadata.get("heading")
                        }
                    )
                )

                current = [sentence]
                current_words = sentence_words

            else:

                current.append(sentence)
                current_words += sentence_words

        if current:

            combined = " ".join(current)

            sub_chunks.append(
                Document(
                    text=combined,
                    metadata={
                        "heading": chunk.metadata.get("heading")
                    }
                )
            )

        return sub_chunks


    #STEP 8: LLaMA SUMMARIZATION (LIMIT LENGTH) 
    def llama_summarize(text: str, max_words: int = 60) -> str:
       
        prompt = f"""
        You are generating narration for an educational video.

        STRICT RULES:
        - Output ONLY narration
        - Do NOT explain the task
        - Do NOT say:
        "This text explains"
        "Here is a summary"
        "The text discusses"
        "This content describes"
        - Speak directly about the topic
        - Maximum {max_words} words
        - Natural spoken narration
        - No markdown
        - No bullet points
        - No introductions

        TEXT:
        {text}
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
                timeout=120
            )
            data = response.json()
            summary = data.get("response", text[:300]).strip()

            words = summary.split()
            if len(words) > max_words:
                summary = " ".join(words[:max_words])
            return summary

        except Exception as e:
            print("Ollama summarization failed:", e)
            return " ".join(text.split()[:max_words])
            
    #  STEP 9: VISUAL PROMPT + Narrative

    def generate_visual_prompt(summary_text: str, heading: str) -> str:

        return (
            f"{heading}, {summary_text}, "
            "photorealistic scene, real world environment, "
            "cinematic lighting, ultra detailed, natural colors"
        ).strip()
    
    def clean_visual_prompt(p):

        if not p:
            return ""

        p = p.replace("\n", " ")
        p = p.replace('"', '')

        p = re.sub(r"\s+", " ", p)

        p = re.sub(r"[-*•]", "", p)

        p = re.sub(r",+", ",", p)

        p = re.sub(r"\.{2,}", ".", p)

        p = re.sub(r'^[,.\s]+', '', p)

        p = p.strip()

        if not p.endswith("."):
            p += "."

        return p
    import re

    def clean_nar_text(text: str) -> str:

        if not text:
            return ""

        text = text.replace("\\n", " ")
        text = text.replace("\\t", " ")

        text = text.replace('\\"', '"')
        text = text.replace("\\", "")

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r'^[,.:;!?()\-\s]+', '', text)

        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\,{2,}', ',', text)

        text = re.sub(
            r'\b(and|or|but|because|which|that|such as)\s*$',
            '',
            text,
            flags=re.IGNORECASE
        )

        words = text.split()

        if (
            len(words) > 3
            and words[0].islower()
            and len(words[0]) <= 4
        ):
            words = words[1:]

        text = " ".join(words)

        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text.strip()
    
    def chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    


    def build_global_narrative(summaries: List[str]) -> str:

        def is_valid_narrative(text: str) -> bool:
            required = ["Intro:", "Explanation:", "Conclusion:"]
            return (
                text
                and len(text.strip()) > 50
                and all(r in text for r in required)
            )

        def safe_generate(prompt: str, temperature=0.2, timeout=120):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature
                        }
                    },
                    timeout=timeout
                )

                data = response.json()
                response = data.get("response", "").strip()
                response = clean_nar_text(response)

                return response
            except Exception as e:
                print("Generation failed:", e)
                return ""
            
        summary_groups = list(chunk_list(summaries, 8))
        mini_narratives = []

        for idx, group in enumerate(summary_groups):

            combined = "\n\n".join(group)

            mini_prompt = f"""
    Create a SHORT structured educational narration.

    STRICT RULES:
    - Use labels exactly:
    Intro:
    Explanation:
    Example:
    Conclusion:
    - No bullet points
    - No markdown
    - Keep it concise

    CONTENT:
    {combined}
    """

            response = safe_generate(mini_prompt, temperature=0.3)

            if not is_valid_narrative(response):
                print(f"Mini narrative {idx+1} invalid. Using raw combined text.")
                response = combined

            mini_narratives.append(response)

        final_combined = "\n\n".join(mini_narratives)

        main_prompt = f"""
    Combine these structured narrations into ONE cohesive educational video narration.

    STRICT RULES:
    - MUST use labels exactly:
    Intro:
    Explanation:
    Example:
    Conclusion:
    - No markdown
    - No bullet points
    - No meta commentary
    - No phrases like:
    "This text explains"
    "Here is the narration"
    - Make transitions natural
    - Keep educational flow smooth

    CONTENT:
    {final_combined}
    """

        narrative = safe_generate(
            main_prompt,
            temperature=0.3
        )

        if is_valid_narrative(narrative):
            return narrative

        print("Main narrative invalid. Retrying with stricter fallback...")

        fallback_prompt = f"""
    Rewrite this into a clean educational narration.

    STRICT FORMAT:

    Intro:
    Explanation:
    Example:
    Conclusion:

    STRICT RULES:
    - Use labels exactly
    - No markdown
    - No bullet points
    - No extra commentary
    - Keep sections clear and readable

    CONTENT:
    {final_combined}
    """

        narrative = safe_generate(
            fallback_prompt,
            temperature=0.0
        )

        if is_valid_narrative(narrative):
            return narrative

        print("Fallback narrative also invalid.")

        deterministic_fallback = f"""
    Intro:
    This video introduces the topic and its main concepts.

    Explanation:
    {final_combined[:2500]}

    Example:
    Examples and applications are discussed throughout the explanation.

    Conclusion:
    The topic has been summarized with its important ideas and practical understanding.
    """

        return deterministic_fallback.strip()
    target_scenes=0
    def parse_narrative_to_scenes(narrative: str):
        total_summaries = len(summaries)

        if total_summaries <= 10:
            target_scenes = 4

        elif total_summaries <= 25:
            target_scenes = 6

        elif total_summaries <= 45:
            target_scenes = 8

        else:
            target_scenes =10

        scenes = []

        cleaned = re.sub(r"\*\*(Intro|Explanation|Example|Conclusion)\*\*", r"\1:", narrative)
        cleaned = re.sub(
    r"(?m)^(Intro|Explanation|Example|Conclusion)\s*$",
    r"\1:",
    cleaned
)

        pattern = (
    r"(Intro|Explanation|Example|Conclusion)"
    r"\s*[:\-]?\s*"
    r"(.*?)"
    r"(?=(Intro|Explanation|Example|Conclusion)\s*[:\-]?|$)"
        )
        matches = re.findall(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        scene_number = 1  
        if not matches:

            print("No structured sections found. Falling back to single scene.")

            narration = narrative.strip()

            return [{
                "scene_id": "scene_001",
                "scene_number": 1,
                "scene_type": "Explanation",
                "heading": "Overview",
                "narration_text": narration,

                "visual_prompt": clean_visual_prompt(
                    extract_visual_keywords(narration)
                ),

                "duration": estimate_duration(narration),

                "image_path": "",
                "audio_path": "",
                "video_path": "",
                "status": "pending"
            }]

        

        for match in matches:
            scene_type = match[0].capitalize()
            text = match[1].strip()
            if not text:
                continue
    
            
            if estimate_duration(text) > 60:
                sentences = re.split(r'(?<=[.!?]) +', text)

                num_parts = min(2, math.ceil(estimate_duration(text) / 20))
                chunk_size = math.ceil(len(sentences) / num_parts)

                parts = []
                for i in range(0, len(sentences), chunk_size):
                    parts.append(" ".join(sentences[i:i + chunk_size]))
            else:
                parts = [text]

            for i, part in enumerate(parts):
                duration = estimate_duration(part)

                base_heading = f"{scene_type} – {infer_topic(text)}"
                heading_final = f"{base_heading} (Part {i+1})" if len(parts) > 1 else base_heading
                
                visual_base = extract_visual_keywords(part)
                visual_prompt = (
                            f"{visual_base}, "
                            "photorealistic, ultra realistic, cinematic lighting, "
                            "natural colors, detailed textures"
                        ).strip() 
                #visual_prompt = clean_visual_prompt(visual_base)
                scenes.append({
                    "scene_id": f"scene_{scene_number:03d}",
                    "scene_number": scene_number,
                    "scene_type": scene_type,
                    "heading": heading_final,
                    "narration_text": part,
                    "visual_prompt": visual_prompt,
                    "duration": duration,
                    "image_path": "",
                    "audio_path": "",
                    "video_path": "",
                    "status": "pending"
                })

                scene_number += 1
        return scenes, target_scenes
    def reduce_scene_count(scenes, max_scenes=4, max_duration=60):

        max_scenes = max(1, max_scenes)

        while len(scenes) > max_scenes:

            best_idx = None
            best_size = float("inf")

            for i in range(len(scenes) - 1):

                merged_text = (
                    scenes[i]["narration_text"]
                    + " "
                    + scenes[i + 1]["narration_text"]
                )

                merged_duration = estimate_duration(merged_text)

                if merged_duration > max_duration:
                    continue

                penalty = (
                    0
                    if scenes[i]["scene_type"] == scenes[i + 1]["scene_type"]
                    else 1000
                )

                merged_size = len(merged_text.split()) + penalty

                if merged_size < best_size:
                    best_size = merged_size
                    best_idx = i

            if best_idx is None:
                break

            scenes[best_idx]["narration_text"] += (
                " " + scenes[best_idx + 1]["narration_text"]
            )

            scenes[best_idx]["duration"] = estimate_duration(
                scenes[best_idx]["narration_text"]
            )

            del scenes[best_idx + 1]

        while len(scenes) > max_scenes and len(scenes) > 1:

            scenes[-2]["narration_text"] += (
                " " + scenes[-1]["narration_text"]
            )

            scenes[-2]["duration"] = estimate_duration(
                scenes[-2]["narration_text"]
            )

            del scenes[-1]

        return scenes


        for idx, scene in enumerate(scenes, start=1):
            scene["scene_number"] = idx
            scene["scene_id"] = f"scene_{idx:03d}"

        return scenes

    def extract_visual_keywords(text: str) -> str:

        prompt = f"""
    Convert the following narration into a CINEMATIC image prompt.

    STRICT RULES:
    - Maximum 30 words
    - Focus on ONE main subject
    - Describe environment + action
    - No storytelling, no multiple paragraphs
    - No explanations
    - No camera/lens terms
    - Avoid showing hands prominently
    - Keep hands relaxed or partially visible
    - Face should be focused
    - Avoid detailing complex scenes
    - Don't use words like close-up, wide, over the shoulder, medium shots
    - Avoid ambiguous roles like "security officer", "manager", "user"
    - Don't keep it too short
    - It needs to explain the environment fully
    - It should be 15-30 words
    STYLE:
    - Focus on a clear, simple visual
    - Prefer social media UI, system screens, or human interaction
    - Avoid complex actions (no typing, hovering, gestures)
    - No drawing grids, No flowcharts. 
    - If mentioning dashboards, always specify "on a computer screen" or "software interface"
    - Avoid generic words like "dashboard" without context
    GOOD EXAMPLES:
    - a user adjusting privacy settings on a social media app
    - a dashboard showing user roles and access levels
    - a login screen with secure authentication

    BAD EXAMPLES:
    - a screen showing login, dashboard, settings, and alerts
    - a user clicking, scrolling, selecting multiple options

    Text:
    {text}
    """
        
        raw = generate_with_llama(prompt)

        cleaned = raw.strip().replace('"', '')
        cleaned = cleaned.replace("\n", " ")   
        cleaned = " ".join(cleaned.split())   

        return cleaned


    BAD_PATTERNS = [
    "please paste",
    "you haven't provided",
    "i'm happy to help",
    "provide the text",
    ]

    def is_bad_summary(text):
        t = text.lower()
        return any(p in t for p in BAD_PATTERNS)
    
    def enforce_max_duration(scenes, max_duration=60):

        final_scenes = []

        for scene in scenes:

            duration = estimate_duration(scene["narration_text"])

            if duration <= max_duration:
                final_scenes.append(scene)
                continue

            sentences = re.split(
                r'(?<=[.!?]) +',
                scene["narration_text"]
            )

            current = ""

            for sent in sentences:

                test = current + " " + sent

                if estimate_duration(test) > max_duration:

                    final_scenes.append({
                        **scene,
                        "narration_text": current.strip(),
                        "duration": estimate_duration(current)
                    })

                    current = sent

                else:
                    current += " " + sent

            if current.strip():
                final_scenes.append({
                    **scene,
                    "narration_text": current.strip(),
                    "duration": estimate_duration(current)
                })


        for idx, scene in enumerate(final_scenes, start=1):
            scene["scene_number"] = idx
            scene["scene_id"] = f"scene_{idx:03d}"

        return final_scenes
    
    from llama_index.core.schema import Document



    if __name__ == "__main__":
        pdf_path = r"D:/Downloads/Structured Document-to-Video Generator with Multilingual .docx"

        ensure_ollama_running()
        ensure_model_exists()
        structured_docs = extract_structured_sections(pdf_path)
        print(f"\nRaw structural sections: {len(structured_docs)}")

        structured_docs = merge_small_sections(structured_docs)
        print(f"After merging small sections: {len(structured_docs)}")

        semantic_chunks = []

        for doc in structured_docs:

            if len(doc.text.split()) < SEMANTIC_SPLIT_MIN_WORDS:
                semantic_chunks.append(doc)

            else:
                semantic_chunks.extend(
                    semantic_splitter.get_nodes_from_documents([doc])
                )

        print(f"\nChunk count: {len(semantic_chunks)}")

        final_chunks_split = []

        for chunk in semantic_chunks:

            final_chunks_split.extend(
                split_large_chunk(chunk, max_words=200)
            )

        print(f"\nFinal chunk count: {len(final_chunks_split)}")

        MAX_REASONABLE_CHUNKS = 60

        if len(final_chunks_split) <= MAX_REASONABLE_CHUNKS:

            print("\nUsing semantic chunk pipeline")

            final_chunks = final_chunks_split

        else:

            print("\nChunk explosion detected")
            print("Falling back to merged structural sections")

            final_chunks = []

            for chunk in final_chunks_split:
                if len(chunk.text.split()) <= 220:
                    final_chunks.append(chunk)

                if len(final_chunks) >= MAX_REASONABLE_CHUNKS:
                    break


        print(f"\nChunks selected for summarization: {len(final_chunks)}")

        print("\nGenerating summaries with LLaMA...\n")
        summaries = []
        for i, chunk in enumerate(final_chunks, 1):
            clean_text = sanitize_text(chunk.text)
            if len(clean_text.split()) < 25:
                continue
            print(f"Accepted summaries: {len(summaries)}")
            summary = llama_summarize(clean_text)
            if not is_bad_summary(summary):
                summaries.append(summary)
            
            print(f"SubChunk {i} | Heading: {chunk.metadata.get('heading')} ")
            print("Preview:", summary[:200], "\n")

        narrative = build_global_narrative(summaries)

        print("\n GLOBAL NARRATIVE (Preview)\n")
        print(narrative[:500]) 
            
        storyboard, target_scenes = parse_narrative_to_scenes(narrative)
        MAX_SCENES = 15

        storyboard = reduce_scene_count(
            storyboard,
            max_scenes=target_scenes
        )
        storyboard = enforce_max_duration(storyboard, 60)
        for scene in storyboard:

            scene["heading"] = clean_nar_text(
                scene["heading"]
            )

            scene["narration_text"] = clean_nar_text(
                scene["narration_text"]
            )

            scene["visual_prompt"] = clean_visual_prompt(
                scene["visual_prompt"]
            )

        #filename = f"storyboard_{int(time.time())}.json"
        for idx, scene in enumerate(storyboard, start=1):

            scene["scene_number"] = idx
            scene["scene_id"] = f"scene_{idx:03d}"

        
        filename = f"storyboard.json"
        save_storyboard(storyboard, out_path=filename)

        
        for s in storyboard[:3]:
            print(f"\n Scene {s['scene_number']}")
            print("Heading:", s['heading'])
            print("Scene Type:", s['scene_type'])
            print("Narration (preview):", s['narration_text'][:150], "...")


except KeyboardInterrupt:
    print("\nExiting prompt. User interruption")

except Exception as e:
    print(f"\nUnexpected error: {e}")

finally:
    print("Done")
    sys.exit(0)