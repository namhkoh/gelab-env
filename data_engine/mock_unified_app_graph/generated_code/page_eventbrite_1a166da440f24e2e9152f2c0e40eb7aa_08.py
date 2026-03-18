# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_08
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10.png
# step_index: 8/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI canvas (1440x2960)
# Available: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Full white background (canvas starts white, but ensure fill)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~72px) - light grey bar
STATUS_BAR_H = 72
draw.rectangle([(0, 0), (1440, STATUS_BAR_H)], fill=(209, 209, 209))

# Subtle bottom edge for status bar (soft divider)
for i in range(4):
    y = STATUS_BAR_H + i
    grey = 210 + i  # gradually slightly darker
    draw.line([(0, y), (1440, y)], fill=(grey, grey, grey))

# Header area (below status bar) - keep white but add subtle separation/shadow
HEADER_TOP = STATUS_BAR_H
HEADER_BOTTOM = 200
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill=(255, 255, 255))

# Soft shadow under header (multi-line subtle)
for i in range(6):
    y = HEADER_BOTTOM + i
    shade = 240 + i*2
    draw.line([(24, y), (1416, y)], fill=(shade, shade, shade))

# Card-like rounded background behind the date selection group
card_left = 32
card_top = 240
card_right = 1408
card_bottom = 720
card_radius = 24
card_fill = (250, 247, 255)  # very subtle lavender tint
card_outline = (230, 220, 235)
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill=card_fill,
    outline=card_outline,
    width=2
)

# Internal subtle divider separating Start Date and End Date areas inside the card
sep_y = 420
draw.line([(card_left + 24, sep_y), (card_right - 24, sep_y)], fill=(235, 232, 240), width=2)

# Add a faint vertical guideline at left to echo layout spacing (non-intrusive)
guide_x = 48
for i in range(card_top, card_bottom, 6):
    draw.line([(guide_x, i), (guide_x, i+3)], fill=(245, 245, 247))

# Large content area background (main content region) - keep white but define subtle grid banding
content_top = card_bottom + 24
content_bottom = 2640
band_height = 160
band_col_light = (255, 255, 255)
band_col_alt = (250, 249, 252)
y = content_top
toggle = False
while y < content_bottom:
    col = band_col_alt if toggle else band_col_light
    draw.rectangle([(0, y), (1440, min(y + band_height, content_bottom))], fill=col)
    y += band_height
    toggle = not toggle

# Top divider above the bottom action area (keeps space visual separation above button region)
divider_y = 2720
draw.line([(24, divider_y), (1440 - 24, divider_y)], fill=(225, 222, 228), width=2)

# Rounded container hint above the bottom (a subtle border to frame where the action button will appear)
hint_left = 48
hint_right = 1392
hint_top = 2728
hint_bottom = 2876  # keep this below the detected button area to avoid drawing over it; this is just a faint frame
hint_radius = 12
draw.rounded_rectangle([(hint_left, hint_top), (hint_right, hint_bottom)],
                       radius=hint_radius,
                       outline=(220, 210, 225),
                       width=2)

# Decorative accent: small left accent dot near header (non-icon, non-text)
accent_center = (120, HEADER_TOP + (HEADER_BOTTOM - HEADER_TOP)//2)
draw.ellipse([(accent_center[0]-6, accent_center[1]-6), (accent_center[0]+6, accent_center[1]+6)], fill=(100, 55, 130))

# Done - structural elements drawn. (Detected icons/text will be pasted on top.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/01_icon_5.31.png
try:
    _c1 = get_crop(1, 58, 65)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["5.31"] = [114, 2, 172, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/02_icon_5.31.png
try:
    _c2 = get_crop(2, 57, 63)
    canvas.paste(_c2, (182, 2), _c2)
except Exception:
    pass
layout["5.31"] = [182, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 61)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 59)
    canvas.paste(_c4, (249, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 6, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/05_icon_5.31.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (12, 72), _c5)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 45, 66)
    canvas.paste(_c6, (1325, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/07_icon_5.31.png
try:
    _c7 = get_crop(7, 89, 62)
    canvas.paste(_c7, (16, 3), _c7)
except Exception:
    pass
layout["5.31"] = [16, 3, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 67, 67)
    canvas.paste(_c8, (1214, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1214, 0, 1281, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/09_icon_What_date.png
try:
    _c9 = get_crop(9, 318, 72)
    canvas.paste(_c9, (558, 112), _c9)
except Exception:
    pass
layout["What_date?"] = [558, 112, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 68)
    canvas.paste(_c10, (1257, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1257, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 63)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 589, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/13_text_End_Date.png
try:
    _c13 = get_crop(13, 253, 67)
    canvas.paste(_c13, (45, 437), _c13)
except Exception:
    pass
layout["End_Date"] = [45, 437, 298, 504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_08_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-10/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
