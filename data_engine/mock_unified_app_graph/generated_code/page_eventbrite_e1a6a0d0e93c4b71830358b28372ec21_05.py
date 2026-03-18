# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_05
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7.png
# step_index: 5/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural UI elements for mobile page
# available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_offwhite = (250, 250, 252)       # very light off-white background
status_gray = (189, 189, 189)       # status bar gray
divider_gray = (200, 193, 205)      # subtle purple-gray divider
muted_card = (246, 249, 255)        # very light blue card background
soft_border = (230, 228, 235)       # subtle border color

# Fill full background (canvas may already be white)
draw.rectangle([0, 0, w, h], fill=bg_offwhite)

# Top status bar area (leave icons to be pasted on top)
status_height = 88
draw.rectangle([0, 0, w, status_height], fill=status_gray)

# Header area / toolbar (below status bar) - keep it subtle and light
header_top = status_height
header_bottom = 220
# Slightly lighter than status bar to separate visually
draw.rectangle([0, header_top, w, header_bottom], fill=bg_offwhite)

# Thin bottom divider under the search/header region
divider_y = 330
draw.line([(40, divider_y), (w - 40, divider_y)], fill=divider_gray, width=3)

# "Options" area card (behind Nearby / Online event groups)
options_top = 380
options_bottom = 560
options_left = 32
options_right = w - 32
options_radius = 28
draw.rounded_rectangle([options_left, options_top, options_right, options_bottom],
                       radius=options_radius,
                       fill=muted_card,
                       outline=soft_border)

# Internal subtle separator within the options card (to imply two groupings)
sep_y = options_top + (options_bottom - options_top) * 0.55
draw.line([(options_left + 30, sep_y), (options_right - 30, sep_y)], fill=soft_border, width=1)

# "Browsing in" content area background (a large subtle panel behind the location selection)
browse_top = 700
browse_bottom = 980
browse_left = 30
browse_right = w - 30
browse_radius = 18
# Keep panel extremely subtle (almost white) but with a soft outline to imply a selectable area
draw.rounded_rectangle([browse_left, browse_top, browse_right, browse_bottom],
                       radius=browse_radius,
                       fill=(255, 255, 255),
                       outline=soft_border,
                       width=1)

# Subtle bottom separator line below the browsing card
draw.line([(40, browse_bottom + 18), (w - 40, browse_bottom + 18)], fill=divider_gray, width=1)

# Light vertical padding guides (non-intrusive thin lines) to visually structure spacing
# (These are very faint and intended purely for layout structure, not content)
guide_color = (245, 245, 247)
draw.line([(options_left + 20, options_top + 12), (options_left + 20, options_bottom - 12)], fill=guide_color, width=1)
draw.line([(options_right - 20, options_top + 12), (options_right - 20, options_bottom - 12)], fill=guide_color, width=1)

# Small bottom shadow for the options card to lift it slightly
shadow_top = options_bottom
shadow_height = 18
shadow = Image.new("RGBA", (w, shadow_height), (0, 0, 0, 0))
sd = Image.new("RGBA", (w, shadow_height), (0,0,0,0))
# draw a soft gradient-like shadow using semi-transparent rectangles
shadow_draw = Image.new("RGBA", (w, shadow_height))
# We cannot import additional modules, but we can draw a few semi-transparent lines onto the canvas directly:
for i in range(shadow_height):
    alpha = int(10 * (1 - i / shadow_height))  # fading alpha
    # compositing by drawing directly onto main canvas with reduced opacity simulated by blending color
    draw.rectangle([options_left, shadow_top + i, options_right, shadow_top + i], fill=(0, 0, 0, alpha))

# Note: icons and texts will be overlaid later at their exact detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 60)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/02_icon_5.18.png
try:
    _c2 = get_crop(2, 59, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["5.18"] = [180, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/03_icon_5.18.png
try:
    _c3 = get_crop(3, 57, 64)
    canvas.paste(_c3, (116, 1), _c3)
except Exception:
    pass
layout["5.18"] = [116, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/04_icon_5.18.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["5.18"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 59)
    canvas.paste(_c6, (1322, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/10_icon_5.18.png
try:
    _c10 = get_crop(10, 93, 63)
    canvas.paste(_c10, (14, 2), _c10)
except Exception:
    pass
layout["5.18"] = [14, 2, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_05_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-7/17_text_Chicago.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 816, 1440, 954]
