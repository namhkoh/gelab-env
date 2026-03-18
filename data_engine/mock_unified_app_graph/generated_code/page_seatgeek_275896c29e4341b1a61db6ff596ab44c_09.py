# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_09
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12.png
# step_index: 9/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the mobile UI page
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw) objects.

w, h = canvas.size

# Colors
bg_white = (255, 255, 255)
light_grey = (243, 244, 245)   # very light grey for status bar / subtle areas
divider = (230, 230, 230)      # thin dividers
card_shadow = (240, 240, 240)  # faint shadow line
muted_bg = (250, 250, 250)

# Fill overall background (canvas starts white, but ensure consistent fill)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# Status bar area (top ~88px) - subtle light bar behind status icons
status_bar_h = 88
draw.rectangle([(0, 0), (w, status_bar_h)], fill=light_grey)

# Top toolbar divider (thin line under status bar)
draw.line([(0, status_bar_h), (w, status_bar_h)], fill=divider, width=1)

# Main content card (rounded) that sits below the large image area.
# The app's photo/crop will be pasted at the very top (0..1084), so start card at 1084.
card_margin_x = 48
card_top = 1084
card_left = card_margin_x
card_right = w - card_margin_x
card_bottom = h - 40
card_radius = 18

# Card background
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill=bg_white,
    outline=None
)

# Subtle top border/shadow for card
draw.line(
    [(card_left + 8, card_top), (card_right - 8, card_top)],
    fill=card_shadow,
    width=1
)

# Horizontal separators within the card to delineate sections.
# Positions chosen to match typical spacing seen in the UI.
separators_y = [
    card_top + 120,   # after price/header area
    card_top + 200,   # after small info
    card_top + 360,   # between delivery/event block and ticket options
    card_top + 520,   # between e-tickets and payment block
    card_top + 700,   # between payment/contact and code block
    card_top + 860    # lower section divider before bottom area
]

for y in separators_y:
    draw.line(
        [(card_left + 24, y), (card_right - 24, y)],
        fill=divider,
        width=1
    )

# Light grouping backgrounds for list blocks - subtle rounded strips (no icons/text)
block_x0 = card_left + 8
block_x1 = card_right - 8
# Example group rectangles (white fill, slight muted background for alternation)
group_blocks = [
    (card_top + 16, card_top + 140),
    (card_top + 152, card_top + 340),
    (card_top + 372, card_top + 500),
    (card_top + 532, card_top + 680),
    (card_top + 712, card_top + 840),
]

for i, (y0, y1) in enumerate(group_blocks):
    # alternate fill slight tint to visually separate groups
    fill = bg_white if (i % 2 == 0) else muted_bg
    draw.rounded_rectangle(
        [(block_x0, y0), (block_x1, y1)],
        radius=12,
        fill=fill,
        outline=None
    )
    # inner left padding alignment guide (thin vertical accent line subtle)
    accent_x = block_x0 + 0
    draw.line([(accent_x, y0 + 6), (accent_x, y1 - 6)], fill=card_shadow, width=1)

# Bottom area faint divider above screen bottom
bottom_div_y = card_bottom - 120
draw.line([(card_left + 12, bottom_div_y), (card_right - 12, bottom_div_y)], fill=divider, width=1)

# Small top-left/back and top-right/share overlay backgrounds are part of image overlay;
# draw subtle translucent toolbar background strip across the top of the image so icons pasted later sit on it.
toolbar_strip_h = 160
toolbar_strip_color = (255, 255, 255, 40)  # alpha ignored here; use very light solid instead
# Use a very light semi-opaque look by picking a slightly off-white
draw.rectangle([(0, status_bar_h), (w, toolbar_strip_h)], fill=(250, 250, 250))

# A subtle full-width thin separator under the toolbar strip to separate image and content below
draw.line([(0, toolbar_strip_h), (w, toolbar_strip_h)], fill=divider, width=1)

# Top large image area bottom fade: draw a faint gradient-like band (simulated with translucent lines)
# Simulate a fade by drawing several horizontal lines with increasing lightness.
fade_top = 980
fade_bottom = card_top
steps = 8
for i in range(steps):
    fy = fade_top + (fade_bottom - fade_top) * (i / float(steps))
    alpha_shade = 245 + int((i / float(steps)) * 10)
    draw.line([(0, fy), (w, fy)], fill=(alpha_shade, alpha_shade, alpha_shade), width=1)

# Final thin edge outlines for the main card to give subtle structure
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    outline=divider,
    width=1
)

# NOTE: Do not draw any icons or text — those elements will be pasted on top automatically.
# This file intentionally only draws background fills, card shapes, separators and structure.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/00_icon_Share.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1272, 84), _c0)
except Exception:
    pass
layout["Share"] = [1272, 84, 1416, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/01_icon_7.50.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (18, 84), _c1)
except Exception:
    pass
layout["7.50"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/02_icon_Tickets_will_be_available_within_minutes.png
try:
    _c2 = get_crop(2, 1260, 252)
    canvas.paste(_c2, (90, 1573), _c2)
except Exception:
    pass
layout["Tickets_will_be_available"] = [90, 1573, 1350, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/03_icon_Angel_Stadium_of_Anaheim_Anaheim_CA.png
try:
    _c3 = get_crop(3, 1260, 252)
    canvas.paste(_c3, (90, 1573), _c3)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim,"] = [90, 1573, 1350, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 55, 70)
    canvas.paste(_c4, (1152, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1152, 0, 1207, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 71, 79)
    canvas.paste(_c5, (307, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [307, 0, 378, 79]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 63, 64)
    canvas.paste(_c6, (1210, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1210, 0, 1273, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/07_icon_1-8_tickets.png
try:
    _c7 = get_crop(7, 1260, 252)
    canvas.paste(_c7, (90, 1573), _c7)
except Exception:
    pass
layout["1-8_tickets"] = [90, 1573, 1350, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 60, 65)
    canvas.paste(_c8, (1314, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1314, 0, 1374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/09_icon_Add_payment_method.png
try:
    _c9 = get_crop(9, 1260, 169)
    canvas.paste(_c9, (90, 2398), _c9)
except Exception:
    pass
layout["Add_payment_method"] = [90, 2398, 1350, 2567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 67, 72)
    canvas.paste(_c10, (238, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [238, 0, 305, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/11_icon_Add_contact_info.png
try:
    _c11 = get_crop(11, 1260, 169)
    canvas.paste(_c11, (90, 2567), _c11)
except Exception:
    pass
layout["Add_contact_info"] = [90, 2567, 1350, 2736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 65)
    canvas.paste(_c12, (1268, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1268, 0, 1313, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/13_icon_S18_each_incl_fees.png
try:
    _c13 = get_crop(13, 1440, 1084)
    canvas.paste(_c13, (0, 0), _c13)
except Exception:
    pass
layout["S18_each,_incl:_fees"] = [0, 0, 1440, 1084]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/14_icon_Expand_image_to_fullscreen.png
try:
    _c14 = get_crop(14, 96, 96)
    canvas.paste(_c14, (60, 928), _c14)
except Exception:
    pass
layout["Expand_image_to_fullscree"] = [60, 928, 156, 1024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/15_icon_Amazing_Deal.png
try:
    _c15 = get_crop(15, 421, 67)
    canvas.paste(_c15, (131, 1221), _c15)
except Exception:
    pass
layout["Amazing_Deal"] = [131, 1221, 552, 1288]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/16_icon_7.50.png
try:
    _c16 = get_crop(16, 55, 66)
    canvas.paste(_c16, (181, 0), _c16)
except Exception:
    pass
layout["7.50"] = [181, 0, 236, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 128, 142)
    canvas.paste(_c17, (1201, 2229), _c17)
except Exception:
    pass
layout["icon_17"] = [1201, 2229, 1329, 2371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/18_icon_Angel_Stadium_of_Anaheim_Anaheim_CA.png
try:
    _c18 = get_crop(18, 1260, 193)
    canvas.paste(_c18, (90, 2205), _c18)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim,"] = [90, 2205, 1350, 2398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 79, 112)
    canvas.paste(_c19, (1217, 2769), _c19)
except Exception:
    pass
layout["icon_19"] = [1217, 2769, 1296, 2881]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/20_icon_7.50.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (18, 84), _c20)
except Exception:
    pass
layout["7.50"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/21_text_Includes_S9_in_fees_per_ticket.png
try:
    _c21 = get_crop(21, 1260, 252)
    canvas.paste(_c21, (90, 1573), _c21)
except Exception:
    pass
layout["Includes_S9_in_fees_per_t"] = [90, 1573, 1350, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/22_text_Have_a_code.png
try:
    _c22 = get_crop(22, 311, 52)
    canvas.paste(_c22, (278, 2794), _c22)
except Exception:
    pass
layout["Have_a_code?"] = [278, 2794, 589, 2846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_09_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-12/23_clickable_Have_a_code.png
try:
    _c23 = get_crop(23, 1260, 169)
    canvas.paste(_c23, (90, 2736), _c23)
except Exception:
    pass
layout["Have_a_code?"] = [90, 2736, 1350, 2905]
