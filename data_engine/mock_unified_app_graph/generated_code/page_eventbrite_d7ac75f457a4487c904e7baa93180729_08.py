# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_08
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10.png
# step_index: 8/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Base background (ensure fresh white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top ~72px) - subtle neutral gray
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Thin darker line at bottom of status bar to separate from toolbar
draw.line((0, status_h - 1, 1440, status_h - 1), fill="#B0B0B0", width=1)

# Toolbar / header area (below status bar)
toolbar_top = status_h
toolbar_bottom = status_h + 88  # ~88px tall toolbar
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill="#FFFFFF")

# Subtle bottom divider for the toolbar
draw.line((24, toolbar_bottom - 1, 1440 - 24, toolbar_bottom - 1), fill="#E6E6E9", width=2)

# Card / content group background (rounded rectangle) that will sit behind the list items
card_left = 24
card_right = 1440 - 24
card_top = 220
card_bottom = 1336

# Soft shadow for the card (light gray offset)
shadow_offset = 8
draw.rounded_rectangle(
    (card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset),
    radius=24,
    fill="#F2F2F4",
    outline=None
)

# Card main surface (slightly off-white to separate from page white)
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=24,
    fill="#FFFFFF",
    outline="#EAEAF0"
)

# Inside horizontal padding for separators (aligned with text margins)
sep_x1 = 48
sep_x2 = 1440 - 48

# Separator positions based on detected item blocks (y coordinates)
# These separators sit inside the card to delineate the selectable rows.
separators = [
    234 + 144,   # bottom of "Anytime" block (detected pos 234 size 144)
    414 + 144,   # bottom of "Today"    (pos 414)
    594 + 144,   # bottom of "Tomorrow" (pos 594)
    774 + 144,   # bottom of "This Week" (pos 774)
    954 + 144,   # bottom of "This Weekend" (pos 954)
    1134 + 144   # bottom of "Choose a date..." (pos 1134)
]

# Draw separators as subtle hairlines inside the card only where they fall within the card bounds
for y in separators:
    if card_top + 8 < y < card_bottom - 8:
        draw.line((sep_x1, y, sep_x2, y), fill="#F0F0F2", width=1)

# Additional subtle divider under the header/title area (centered visually)
header_div_y = toolbar_bottom + 24
draw.line((24, header_div_y, 1440 - 24, header_div_y), fill="#FBFBFC", width=1)

# Right-side accent guideline (for alignment of checkmarks/icons) - faint vertical guide
guide_x = 1336
draw.line((guide_x, card_top + 24, guide_x, card_bottom - 24), fill="#F7F7F9", width=1)

# Light bottom area fill to show page continuation
bottom_band_top = card_bottom + 24
draw.rectangle((0, bottom_band_top, 1440, 2960), fill="#FFFFFF")

# Small decorative top-left back area background (behind the back arrow icon area)
back_icon_area = (24, toolbar_top + 18, 96, toolbar_bottom - 18)
draw.rectangle(back_icon_area, fill="#FFFFFF")

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 64, 61)
    canvas.paste(_c0, (308, 3), _c0)
except Exception:
    pass
layout["icon_0"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/01_icon_4.39.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["4.39"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/02_icon_4.39.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (12, 72), _c2)
except Exception:
    pass
layout["4.39"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/03_icon_Anytime.png
try:
    _c3 = get_crop(3, 1344, 144)
    canvas.paste(_c3, (48, 234), _c3)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/04_icon_4.39.png
try:
    _c4 = get_crop(4, 58, 63)
    canvas.paste(_c4, (115, 2), _c4)
except Exception:
    pass
layout["4.39"] = [115, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 59)
    canvas.paste(_c5, (248, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 4, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 61)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/08_icon_4.39.png
try:
    _c8 = get_crop(8, 91, 60)
    canvas.paste(_c8, (16, 4), _c8)
except Exception:
    pass
layout["4.39"] = [16, 4, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_08_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-10/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
