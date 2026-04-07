"""
PDF Chunking Pipeline Visualization — 3Blue1Brown style
Renders the full EDMS RAG pipeline from PDF upload to semantic search.
"""

from manim import *
import numpy as np

# ── Color palette ──────────────────────────────────────────────
BG        = "#0f172a"   # slate-950
BLUE      = "#3b82f6"
CYAN      = "#06b6d4"
GREEN     = "#22c55e"
ORANGE    = "#f97316"
PURPLE    = "#a855f7"
PINK      = "#ec4899"
YELLOW    = "#eab308"
RED       = "#ef4444"
SLATE     = "#94a3b8"
WHITE_T   = "#e2e8f0"


def box(label, color, w=2.8, h=0.7, font_size=22):
    r = RoundedRectangle(
        corner_radius=0.12, width=w, height=h,
        stroke_color=color, fill_color=color, fill_opacity=0.15,
        stroke_width=2,
    )
    t = Text(label, font_size=font_size, color=WHITE_T)
    t.move_to(r.get_center())
    return VGroup(r, t)


def arrow_between(a, b, color=SLATE):
    return Arrow(
        a.get_bottom(), b.get_top(),
        buff=0.1, color=color, stroke_width=2,
        max_tip_length_to_length_ratio=0.15,
    )


def stage_label(text, color, font_size=16):
    return Text(text, font_size=font_size, color=color, slant=ITALIC)


# ═══════════════════════════════════════════════════════════════
# SCENE 1 — High-level pipeline overview
# ═══════════════════════════════════════════════════════════════
class PipelineOverview(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("PDF Chunking Pipeline", font_size=40, color=WHITE_T)
        subtitle = Text("EDMS RAG — End to End", font_size=20, color=SLATE)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.3), run_time=1.2)
        self.wait(0.6)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Pipeline boxes ──
        stages = [
            ("1. Upload PDF",       BLUE),
            ("2. Parse & Extract",  CYAN),
            ("3. Build Structure",  GREEN),
            ("4. Chunk (Semantic)", ORANGE),
            ("5. Enrich Context",   PURPLE),
            ("6. Embed (768-d)",    PINK),
            ("7. Store (pgvector)", RED),
            ("8. Hybrid Search",    YELLOW),
        ]

        boxes = VGroup(*[box(l, c, w=3.2, h=0.65) for l, c in stages])
        boxes.arrange(DOWN, buff=0.28)
        boxes.scale_to_fit_height(6.5)
        boxes.move_to(ORIGIN)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_bottom(), boxes[i + 1].get_top(),
                buff=0.08, color=SLATE, stroke_width=1.8,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows.add(a)

        for i, (b, a_) in enumerate(zip(boxes, [None] + list(arrows))):
            anims = [FadeIn(b, shift=DOWN * 0.2)]
            if a_ is not None:
                anims.append(GrowArrow(a_))
            self.play(*anims, run_time=0.35)

        self.wait(1.5)
        self.play(FadeOut(VGroup(boxes, arrows)))


# ═══════════════════════════════════════════════════════════════
# SCENE 2 — PDF Parsing with fallback chain
# ═══════════════════════════════════════════════════════════════
class ParsingFallback(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Stage 2: Parse & Extract", font_size=34, color=CYAN)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.6)

        # PDF icon
        pdf_box = box("PDF File", BLUE, w=2.2, h=0.6)
        pdf_box.move_to(UP * 2 + LEFT * 4)

        # Parsers — fallback chain
        parsers = [
            ("pymupdf4llm", GREEN,  "Primary — markdown + hierarchy"),
            ("docling",     ORANGE, "Fallback 1 — ML layout-aware"),
            ("unstructured",PURPLE, "Fallback 2 — element classification"),
            ("PyMuPDF",     RED,    "Fallback 3 — font heuristics"),
        ]

        parser_boxes = VGroup()
        desc_labels = VGroup()
        for i, (name, color, desc) in enumerate(parsers):
            b = box(name, color, w=2.6, h=0.55, font_size=20)
            b.move_to(RIGHT * 1 + UP * (1.5 - i * 1.1))
            d = Text(desc, font_size=13, color=SLATE)
            d.next_to(b, RIGHT, buff=0.3)
            parser_boxes.add(b)
            desc_labels.add(d)

        # Fallback arrows
        fallback_arrows = VGroup()
        for i in range(len(parser_boxes) - 1):
            a = Arrow(
                parser_boxes[i].get_bottom(),
                parser_boxes[i + 1].get_top(),
                buff=0.08, color=RED, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            label = Text("fail", font_size=11, color=RED)
            label.next_to(a, LEFT, buff=0.1)
            fallback_arrows.add(VGroup(a, label))

        # Output
        output = box("DocumentStructure", GREEN, w=3.2, h=0.55, font_size=20)
        output.move_to(RIGHT * 1 + DOWN * 2.5)

        success_arrow = Arrow(
            parser_boxes[0].get_right() + RIGHT * 1.2,
            output.get_top(),
            buff=0.15, color=GREEN, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        success_label = Text("success", font_size=12, color=GREEN)
        success_label.next_to(success_arrow, RIGHT, buff=0.1)

        self.play(FadeIn(pdf_box))
        self.wait(0.3)

        pdf_arrow = Arrow(
            pdf_box.get_right(), parser_boxes[0].get_left(),
            buff=0.15, color=SLATE, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        self.play(GrowArrow(pdf_arrow), run_time=0.4)

        for i, (pb, dl) in enumerate(zip(parser_boxes, desc_labels)):
            anims = [FadeIn(pb, shift=RIGHT * 0.2), FadeIn(dl)]
            if i > 0:
                anims.append(GrowArrow(fallback_arrows[i - 1][0]))
                anims.append(FadeIn(fallback_arrows[i - 1][1]))
            self.play(*anims, run_time=0.45)

        self.play(FadeIn(output, shift=UP * 0.2), GrowArrow(success_arrow),
                  FadeIn(success_label), run_time=0.5)

        # Output contents
        fields = [
            "title, doc_type",
            "pages[] → StructuredPage",
            "  sections[] → title, level, content",
            "  section_type: text | table | image | list",
        ]
        field_group = VGroup()
        for f in fields:
            t = Text(f, font_size=13, color=SLATE)
            field_group.add(t)
        field_group.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        field_group.next_to(output, DOWN, buff=0.3)
        self.play(FadeIn(field_group), run_time=0.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# SCENE 3 — Semantic Chunking deep dive
# ═══════════════════════════════════════════════════════════════
class SemanticChunking(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Stage 4: Semantic Chunking", font_size=34, color=ORANGE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.6)

        # ── Show document text as blocks ──
        doc_label = Text("Document sections", font_size=16, color=SLATE)
        doc_label.move_to(UP * 2.5 + LEFT * 3)
        self.play(FadeIn(doc_label))

        section_colors = [BLUE, CYAN, GREEN]
        section_labels = ["Section A\n(Research)", "Section B\n(Methods)", "Section C\n(Results)"]
        sections = VGroup()
        for i, (c, l) in enumerate(zip(section_colors, section_labels)):
            r = RoundedRectangle(
                corner_radius=0.08, width=1.8, height=1.0,
                stroke_color=c, fill_color=c, fill_opacity=0.12,
                stroke_width=2,
            )
            t = Text(l, font_size=12, color=WHITE_T)
            t.move_to(r.get_center())
            sections.add(VGroup(r, t))
        sections.arrange(RIGHT, buff=0.25)
        sections.next_to(doc_label, DOWN, buff=0.3)
        self.play(FadeIn(sections), run_time=0.5)

        # ── Parent-Child explanation ──
        pc_title = Text("Parent-Child Chunking", font_size=20, color=ORANGE)
        pc_title.move_to(DOWN * 0.3)
        self.play(Write(pc_title), run_time=0.4)

        # Parent
        parent = RoundedRectangle(
            corner_radius=0.1, width=5, height=0.7,
            stroke_color=ORANGE, fill_color=ORANGE, fill_opacity=0.1,
            stroke_width=2,
        )
        parent_label = Text("Parent Chunk (full section text — no size limit)", font_size=13, color=ORANGE)
        parent_label.move_to(parent.get_center())
        parent_g = VGroup(parent, parent_label)
        parent_g.next_to(pc_title, DOWN, buff=0.4)

        # Children
        child_colors = [PURPLE, PINK, CYAN]
        children = VGroup()
        for i, cc in enumerate(child_colors):
            r = RoundedRectangle(
                corner_radius=0.08, width=1.5, height=0.55,
                stroke_color=cc, fill_color=cc, fill_opacity=0.12,
                stroke_width=1.5,
            )
            t = Text(f"Child {i+1}\n(512 chars)", font_size=11, color=WHITE_T)
            t.move_to(r.get_center())
            children.add(VGroup(r, t))
        children.arrange(RIGHT, buff=0.2)
        children.next_to(parent_g, DOWN, buff=0.35)

        # Overlap indicator
        overlap_line = DashedLine(
            children[0].get_right() + LEFT * 0.15,
            children[1].get_left() + RIGHT * 0.15,
            color=YELLOW, stroke_width=1.5,
        )
        overlap_label = Text("50 char overlap", font_size=10, color=YELLOW)
        overlap_label.next_to(overlap_line, DOWN, buff=0.08)

        self.play(FadeIn(parent_g), run_time=0.4)
        self.play(FadeIn(children), run_time=0.5)
        self.play(Create(overlap_line), FadeIn(overlap_label), run_time=0.4)

        # ── Config sidebar ──
        config_items = [
            "child_size = 512",
            "overlap = 50",
            "min_child = 90",
            "min_parent = 140",
            "separators: \\n\\n, \\n, '. ', ' '",
        ]
        config_group = VGroup()
        for c in config_items:
            t = Text(c, font_size=12, color=SLATE)
            config_group.add(t)
        config_group.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        config_box = SurroundingRectangle(config_group, color=SLATE, buff=0.15,
                                           corner_radius=0.08, stroke_width=1)
        config_title = Text("Config", font_size=14, color=WHITE_T)
        config_title.next_to(config_box, UP, buff=0.1)
        config_all = VGroup(config_title, config_box, config_group)
        config_all.move_to(RIGHT * 5 + DOWN * 0.5)

        self.play(FadeIn(config_all), run_time=0.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# SCENE 4 — Semantic Splitting visualization (similarity curve)
# ═══════════════════════════════════════════════════════════════
class SemanticSplitting(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Semantic Splitting — Embedding Similarity", font_size=30, color=PURPLE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.6)

        # ── Sentence blocks ──
        sentence_label = Text("Sentences", font_size=16, color=SLATE)
        sentence_label.move_to(UP * 2.2 + LEFT * 4.5)
        self.play(FadeIn(sentence_label))

        sentences = VGroup()
        for i in range(10):
            r = Rectangle(width=0.8, height=0.4,
                          stroke_color=CYAN, fill_color=CYAN, fill_opacity=0.15,
                          stroke_width=1)
            t = Text(f"S{i+1}", font_size=11, color=WHITE_T)
            t.move_to(r.get_center())
            sentences.add(VGroup(r, t))
        sentences.arrange(RIGHT, buff=0.05)
        sentences.next_to(sentence_label, DOWN, buff=0.2)
        sentences.scale_to_fit_width(10)
        self.play(FadeIn(sentences), run_time=0.5)

        # ── Similarity curve ──
        ax = Axes(
            x_range=[0, 9, 1], y_range=[0, 1, 0.25],
            x_length=9, y_length=2.5,
            axis_config={"color": SLATE, "stroke_width": 1, "include_tip": False},
            tips=False,
        )
        ax.move_to(DOWN * 0.7)

        x_label = Text("Sentence pair", font_size=13, color=SLATE)
        x_label.next_to(ax, DOWN, buff=0.2)
        y_label = Text("Cosine Similarity", font_size=13, color=SLATE)
        y_label.rotate(PI / 2)
        y_label.next_to(ax, LEFT, buff=0.3)

        # Simulated similarity values with two dips (topic changes)
        sims = [0.85, 0.78, 0.82, 0.35, 0.88, 0.91, 0.87, 0.28, 0.83]
        points = [ax.c2p(i, s) for i, s in enumerate(sims)]

        # Smooth curve through points
        dots = VGroup(*[Dot(p, radius=0.06, color=PURPLE) for p in points])
        line_segments = VGroup()
        for i in range(len(points) - 1):
            l = Line(points[i], points[i + 1], color=PURPLE, stroke_width=2)
            line_segments.add(l)

        # Threshold line (25th percentile)
        threshold = np.percentile(sims, 25)
        thresh_line = DashedLine(
            ax.c2p(0, threshold), ax.c2p(9, threshold),
            color=RED, stroke_width=1.5,
        )
        thresh_label = Text(f"25th percentile = {threshold:.2f}", font_size=12, color=RED)
        thresh_label.next_to(thresh_line, RIGHT, buff=0.2)

        self.play(Create(ax), FadeIn(x_label), FadeIn(y_label), run_time=0.6)
        self.play(Create(line_segments), FadeIn(dots), run_time=0.8)
        self.play(Create(thresh_line), FadeIn(thresh_label), run_time=0.4)

        # ── Highlight split points ──
        split_indices = [i for i, s in enumerate(sims) if s < threshold]
        split_markers = VGroup()
        for idx in split_indices:
            arr = Arrow(
                ax.c2p(idx, sims[idx]) + UP * 0.4,
                ax.c2p(idx, sims[idx]),
                buff=0.05, color=YELLOW, stroke_width=2,
                max_tip_length_to_length_ratio=0.3,
            )
            lbl = Text("SPLIT", font_size=11, color=YELLOW)
            lbl.next_to(arr, UP, buff=0.05)
            split_markers.add(VGroup(arr, lbl))

        self.play(FadeIn(split_markers), run_time=0.5)

        # Show resulting chunks
        chunk_label = Text("Resulting Chunks:", font_size=16, color=GREEN)
        chunk_label.move_to(DOWN * 2.8 + LEFT * 3)

        chunks_vis = VGroup()
        chunk_ranges = [(0, 3), (4, 6), (7, 9)]
        chunk_colors = [BLUE, GREEN, ORANGE]
        for i, ((start, end), cc) in enumerate(zip(chunk_ranges, chunk_colors)):
            r = RoundedRectangle(
                corner_radius=0.08, width=2.2, height=0.4,
                stroke_color=cc, fill_color=cc, fill_opacity=0.15,
                stroke_width=1.5,
            )
            t = Text(f"Chunk {i+1}: S{start+1}–S{end+1}", font_size=12, color=WHITE_T)
            t.move_to(r.get_center())
            chunks_vis.add(VGroup(r, t))
        chunks_vis.arrange(RIGHT, buff=0.2)
        chunks_vis.next_to(chunk_label, RIGHT, buff=0.3)

        self.play(FadeIn(chunk_label), FadeIn(chunks_vis), run_time=0.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# SCENE 5 — Context Enrichment
# ═══════════════════════════════════════════════════════════════
class ContextEnrichment(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Stage 5: Contextual Enrichment", font_size=32, color=PURPLE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.6)

        # Before
        before_label = Text("Before (raw chunk)", font_size=16, color=RED)
        before_label.move_to(UP * 2 + LEFT * 3)

        raw_text = Text(
            '"The model achieved 94.2%\naccuracy on the test set..."',
            font_size=14, color=WHITE_T,
        )
        raw_box = SurroundingRectangle(raw_text, color=RED, buff=0.15,
                                        corner_radius=0.08, stroke_width=1.5)
        before_group = VGroup(raw_box, raw_text)
        before_group.next_to(before_label, DOWN, buff=0.3)

        # After
        after_label = Text("After (enriched chunk)", font_size=16, color=GREEN)
        after_label.move_to(UP * 2 + RIGHT * 3)

        enriched_text = Text(
            'From "DeepSeek Math" (research_paper),\n'
            'section "Results", page 5.\n'
            'Summary: Evaluation of model performance.\n\n'
            '"The model achieved 94.2%\n'
            'accuracy on the test set..."',
            font_size=12, color=WHITE_T, line_spacing=1.2,
        )
        enriched_box = SurroundingRectangle(enriched_text, color=GREEN, buff=0.15,
                                             corner_radius=0.08, stroke_width=1.5)
        after_group = VGroup(enriched_box, enriched_text)
        after_group.next_to(after_label, DOWN, buff=0.3)

        # Context prefix highlight
        prefix_highlight = SurroundingRectangle(
            enriched_text[:3],  # approximate
            color=PURPLE, buff=0.05, corner_radius=0.05, stroke_width=1,
            fill_color=PURPLE, fill_opacity=0.1,
        )

        arrow = Arrow(
            before_group.get_right(), after_group.get_left(),
            buff=0.3, color=PURPLE, stroke_width=2,
        )
        arrow_label = Text("ContextEnricher", font_size=13, color=PURPLE)
        arrow_label.next_to(arrow, UP, buff=0.1)

        self.play(FadeIn(before_label), FadeIn(before_group), run_time=0.4)
        self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=0.3)
        self.play(FadeIn(after_label), FadeIn(after_group), run_time=0.4)

        # Modes
        modes_title = Text("Enrichment Modes", font_size=18, color=WHITE_T)
        modes_title.move_to(DOWN * 1.5)
        modes = [
            ("template", GREEN,  "Fast, zero-cost — uses doc metadata"),
            ("llm",      ORANGE, "LLM summary + template context"),
            ("full_llm", PURPLE, "LLM for both summary & per-chunk"),
            ("off",      SLATE,  "No enrichment"),
        ]
        mode_group = VGroup()
        for name, color, desc in modes:
            dot = Dot(radius=0.06, color=color)
            lbl = Text(f"{name}: {desc}", font_size=12, color=SLATE)
            lbl.next_to(dot, RIGHT, buff=0.15)
            mode_group.add(VGroup(dot, lbl))
        mode_group.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        mode_group.next_to(modes_title, DOWN, buff=0.25)

        self.play(FadeIn(modes_title), FadeIn(mode_group), run_time=0.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# SCENE 6 — Embedding + Storage
# ═══════════════════════════════════════════════════════════════
class EmbeddingStorage(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Stages 6-7: Embed & Store", font_size=32, color=PINK)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.6)

        # Chunk input
        chunk_in = box("Enriched Chunks", PURPLE, w=2.8, h=0.55)
        chunk_in.move_to(UP * 2.2 + LEFT * 4)

        # Embedding model
        embed_box = box("BAAI/bge-base-en-v1.5", PINK, w=3.5, h=0.6)
        embed_box.move_to(UP * 2.2)
        embed_detail = Text("768-dim vectors | batch=32", font_size=12, color=SLATE)
        embed_detail.next_to(embed_box, DOWN, buff=0.15)

        a1 = Arrow(chunk_in.get_right(), embed_box.get_left(), buff=0.1,
                    color=SLATE, stroke_width=2)

        self.play(FadeIn(chunk_in), GrowArrow(a1), FadeIn(embed_box),
                  FadeIn(embed_detail), run_time=0.5)

        # Vector visualization — show 768-dim as heatmap
        vec_label = Text("768-dimensional embedding", font_size=14, color=PINK)
        vec_label.move_to(UP * 0.6)

        np.random.seed(42)
        vec_vals = np.random.randn(48)  # show 48 as representative
        vec_rects = VGroup()
        for v in vec_vals:
            intensity = (v - vec_vals.min()) / (vec_vals.max() - vec_vals.min())
            c = interpolate_color(ManimColor(BLUE), ManimColor(PINK), intensity)
            r = Rectangle(width=0.17, height=0.35, fill_color=c,
                          fill_opacity=0.8, stroke_width=0)
            vec_rects.add(r)
        vec_rects.arrange(RIGHT, buff=0)
        vec_rects.next_to(vec_label, DOWN, buff=0.2)
        vec_rects.scale_to_fit_width(9)

        ellipsis = Text("... (768 dims total)", font_size=11, color=SLATE)
        ellipsis.next_to(vec_rects, RIGHT, buff=0.15)

        a2 = Arrow(embed_box.get_bottom(), vec_label.get_top(), buff=0.15,
                    color=SLATE, stroke_width=2)

        self.play(GrowArrow(a2), FadeIn(vec_label), run_time=0.3)
        self.play(FadeIn(vec_rects), FadeIn(ellipsis), run_time=0.6)

        # pgvector storage
        pg_box = box("PostgreSQL + pgvector", RED, w=4, h=0.6)
        pg_box.move_to(DOWN * 1.2)

        schema_items = [
            "rag.chunks: id, content, enriched_content,",
            "  embedding vector(768), content_tsv tsvector,",
            "  page_number, chunk_type, parent_id, metadata",
            "",
            "Indexes: HNSW (cosine) + GIN (FTS) + B-tree",
        ]
        schema_group = VGroup()
        for s in schema_items:
            t = Text(s, font_size=11, color=SLATE)
            schema_group.add(t)
        schema_group.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        schema_group.next_to(pg_box, DOWN, buff=0.25)

        a3 = Arrow(vec_rects.get_bottom() + DOWN * 0.1, pg_box.get_top(),
                    buff=0.15, color=SLATE, stroke_width=2)

        self.play(GrowArrow(a3), FadeIn(pg_box), run_time=0.4)
        self.play(FadeIn(schema_group), run_time=0.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# SCENE 7 — Hybrid Search & Retrieval
# ═══════════════════════════════════════════════════════════════
class HybridSearch(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Stage 8: Hybrid Search", font_size=32, color=YELLOW)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.6)

        # Query input
        query_box = box("User Query", BLUE, w=2.4, h=0.55)
        query_box.move_to(UP * 2.5 + LEFT * 4.5)

        # Topic extraction
        topic_box = box("Extract Topic", CYAN, w=2.2, h=0.5, font_size=18)
        topic_box.move_to(UP * 2.5 + LEFT * 1)
        a0 = Arrow(query_box.get_right(), topic_box.get_left(), buff=0.1,
                    color=SLATE, stroke_width=2)

        # Embed query
        embed_q = box("Embed Query", PINK, w=2.2, h=0.5, font_size=18)
        embed_q.move_to(UP * 2.5 + RIGHT * 2.2)
        a0b = Arrow(topic_box.get_right(), embed_q.get_left(), buff=0.1,
                     color=SLATE, stroke_width=2)

        self.play(FadeIn(query_box), GrowArrow(a0), FadeIn(topic_box),
                  GrowArrow(a0b), FadeIn(embed_q), run_time=0.5)

        # Two search paths
        vec_search = box("Vector Search\n(cosine, top 50)", PINK, w=3, h=0.7, font_size=15)
        vec_search.move_to(UP * 0.8 + LEFT * 2.5)

        kw_search = box("Keyword Search\n(ts_rank_cd, top 50)", GREEN, w=3, h=0.7, font_size=15)
        kw_search.move_to(UP * 0.8 + RIGHT * 2.5)

        a1 = Arrow(embed_q.get_bottom(), vec_search.get_top(), buff=0.1,
                    color=PINK, stroke_width=2)
        a2 = Arrow(embed_q.get_bottom(), kw_search.get_top(), buff=0.1,
                    color=GREEN, stroke_width=2)

        self.play(GrowArrow(a1), FadeIn(vec_search),
                  GrowArrow(a2), FadeIn(kw_search), run_time=0.5)

        # RRF Fusion
        rrf_box = box("RRF Fusion (k=60)", YELLOW, w=3.5, h=0.6)
        rrf_box.move_to(DOWN * 0.5)

        a3 = Arrow(vec_search.get_bottom(), rrf_box.get_top() + LEFT * 0.5,
                    buff=0.1, color=SLATE, stroke_width=2)
        a4 = Arrow(kw_search.get_bottom(), rrf_box.get_top() + RIGHT * 0.5,
                    buff=0.1, color=SLATE, stroke_width=2)

        rrf_formula = Text(
            "score(d) = sum( 1 / (k + rank(d)) )",
            font_size=16, color=YELLOW,
        )
        rrf_formula.next_to(rrf_box, RIGHT, buff=0.4)

        self.play(GrowArrow(a3), GrowArrow(a4), FadeIn(rrf_box), run_time=0.4)
        self.play(FadeIn(rrf_formula), run_time=0.4)

        # Reranker
        rerank = box("CrossEncoder Rerank\n(ms-marco-MiniLM, top 15)", ORANGE, w=4, h=0.7, font_size=14)
        rerank.move_to(DOWN * 1.7)

        a5 = Arrow(rrf_box.get_bottom(), rerank.get_top(), buff=0.1,
                    color=SLATE, stroke_width=2)
        self.play(GrowArrow(a5), FadeIn(rerank), run_time=0.4)

        # Output
        output = box("Top-K Pages + Highlights", GREEN, w=3.8, h=0.6)
        output.move_to(DOWN * 3)

        a6 = Arrow(rerank.get_bottom(), output.get_top(), buff=0.1,
                    color=GREEN, stroke_width=2)
        self.play(GrowArrow(a6), FadeIn(output), run_time=0.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ═══════════════════════════════════════════════════════════════
# SCENE 8 — Full Pipeline Summary (animated flow)
# ═══════════════════════════════════════════════════════════════
class FullPipelineSummary(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Complete Pipeline Flow", font_size=36, color=WHITE_T)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # ── Left column: Ingestion ──
        left_title = Text("INGESTION", font_size=18, color=BLUE)
        left_title.move_to(UP * 2.5 + LEFT * 4)

        left_stages = [
            ("PDF Upload", BLUE),
            ("pymupdf4llm Parse", CYAN),
            ("Structure Extract", GREEN),
            ("Semantic Chunk", ORANGE),
            ("Context Enrich", PURPLE),
            ("BGE Embed (768d)", PINK),
            ("pgvector Store", RED),
        ]

        left_boxes = VGroup()
        for l, c in left_stages:
            left_boxes.add(box(l, c, w=2.8, h=0.45, font_size=14))
        left_boxes.arrange(DOWN, buff=0.15)
        left_boxes.next_to(left_title, DOWN, buff=0.2)

        left_arrows = VGroup()
        for i in range(len(left_boxes) - 1):
            a = Arrow(
                left_boxes[i].get_bottom(), left_boxes[i + 1].get_top(),
                buff=0.05, color=SLATE, stroke_width=1.2,
                max_tip_length_to_length_ratio=0.25,
            )
            left_arrows.add(a)

        # ── Right column: Retrieval ──
        right_title = Text("RETRIEVAL", font_size=18, color=YELLOW)
        right_title.move_to(UP * 2.5 + RIGHT * 4)

        right_stages = [
            ("User Query", BLUE),
            ("Topic Extract", CYAN),
            ("Embed Query", PINK),
            ("Vector Search", PINK),
            ("Keyword Search", GREEN),
            ("RRF Fusion", YELLOW),
            ("CrossEncoder Rerank", ORANGE),
            ("Top-K Results", GREEN),
        ]

        right_boxes = VGroup()
        for l, c in right_stages:
            right_boxes.add(box(l, c, w=2.8, h=0.42, font_size=14))
        right_boxes.arrange(DOWN, buff=0.12)
        right_boxes.next_to(right_title, DOWN, buff=0.2)

        right_arrows = VGroup()
        for i in range(len(right_boxes) - 1):
            if i == 2:  # fork to vector + keyword
                continue
            a = Arrow(
                right_boxes[i].get_bottom(), right_boxes[i + 1].get_top(),
                buff=0.05, color=SLATE, stroke_width=1.2,
                max_tip_length_to_length_ratio=0.25,
            )
            right_arrows.add(a)

        # ── Center: shared storage ──
        db_box = box("pgvector\nHNSW Index", RED, w=2.2, h=0.7, font_size=14)
        db_box.move_to(DOWN * 0.5)

        connect_left = Arrow(
            left_boxes[-1].get_right(), db_box.get_left(),
            buff=0.15, color=RED, stroke_width=2,
        )
        connect_right = Arrow(
            db_box.get_right(), right_boxes[3].get_left(),
            buff=0.15, color=PINK, stroke_width=2,
        )

        # Animate left
        self.play(FadeIn(left_title), run_time=0.3)
        for i, b in enumerate(left_boxes):
            anims = [FadeIn(b, shift=DOWN * 0.15)]
            if i > 0:
                anims.append(GrowArrow(left_arrows[i - 1]))
            self.play(*anims, run_time=0.2)

        # Animate right
        self.play(FadeIn(right_title), run_time=0.3)
        for i, b in enumerate(right_boxes):
            anims = [FadeIn(b, shift=DOWN * 0.15)]
            self.play(*anims, run_time=0.18)

        for a in right_arrows:
            self.play(GrowArrow(a), run_time=0.1)

        # Center DB + connections
        self.play(FadeIn(db_box), GrowArrow(connect_left),
                  GrowArrow(connect_right), run_time=0.5)

        # ── Data flow pulse ──
        pulse = Dot(radius=0.1, color=YELLOW, fill_opacity=0.8)
        pulse.move_to(left_boxes[0].get_center())

        path_points = [b.get_center() for b in left_boxes]
        path_points.append(db_box.get_center())

        self.play(FadeIn(pulse), run_time=0.2)
        for p in path_points[1:]:
            self.play(pulse.animate.move_to(p), run_time=0.2)
        self.play(FadeOut(pulse), run_time=0.2)

        # Query pulse
        pulse2 = Dot(radius=0.1, color=CYAN, fill_opacity=0.8)
        pulse2.move_to(right_boxes[0].get_center())
        query_path = [b.get_center() for b in right_boxes]

        self.play(FadeIn(pulse2), run_time=0.2)
        for p in query_path[1:]:
            self.play(pulse2.animate.move_to(p), run_time=0.15)
        self.play(FadeOut(pulse2), run_time=0.2)

        self.wait(2)


# ═══════════════════════════════════════════════════════════════
# COMBINED — renders all scenes sequentially
# ═══════════════════════════════════════════════════════════════
class PDFChunkingPipeline(Scene):
    """Render all scenes as one continuous video."""
    def construct(self):
        scenes = [
            PipelineOverview,
            ParsingFallback,
            SemanticChunking,
            SemanticSplitting,
            ContextEnrichment,
            EmbeddingStorage,
            HybridSearch,
            FullPipelineSummary,
        ]
        for SceneClass in scenes:
            s = SceneClass()
            s.camera = self.camera
            s.mobjects = self.mobjects.copy()
            s.construct()
            # clear for next scene
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.3)
