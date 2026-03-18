# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_07
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9.png
# step_index: 7/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top gradient banner
banner_h = 340
for y in range(banner_h):
    t = y / max(1, banner_h - 1)
    # Blend from warm brownish to soft beige
    r = int(170 * (1 - t) + 235 * t)
    g = int(140 * (1 - t) + 225 * t)
    b = int(125 * (1 - t) + 215 * t)
    draw.line([(0, y), (canvas.width, y)], fill=(r, g, b))

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([0, 0, canvas.width, status_h], fill=(160, 160, 160))

# Soft bottom edge shadow of the banner
draw.rectangle([0, banner_h - 6, canvas.width, banner_h], fill=(210, 200, 195))

# Large white header card (profile area) with subtle drop shadow
card_left, card_top = 60, 220
card_right, card_bottom = canvas.width - 60, 980
card_radius = 28

# shadow layers (fainter and more spread)
for i, alpha_shade in enumerate([(220,220,220),(230,230,230),(240,240,240),(250,250,250)]):
    offset = 10 - i*3
    shadow_box = [card_left, card_top + offset, card_right, card_bottom + offset]
    # darker for lower offsets
    shade = (200 + i*8, 200 + i*8, 200 + i*8)
    draw.rounded_rectangle(shadow_box, radius=card_radius, fill=shade)

# white card on top
draw.rounded_rectangle([card_left, card_top, card_right, card_bottom], radius=card_radius, fill=(255,255,255))

# Divider line under header card (subtle)
divider_y = card_bottom + 6
draw.line([(40, divider_y), (canvas.width - 40, divider_y)], fill=(230,230,235), width=2)

# Tab area background (slightly elevated)
tabs_top = divider_y + 12
tabs_bottom = tabs_top + 140
draw.rectangle([40, tabs_top, canvas.width - 40, tabs_bottom], fill=(255,255,255))

# Thin separator under tabs
sep_y = tabs_bottom + 10
draw.line([(40, sep_y), (canvas.width - 40, sep_y)], fill=(235,235,240), width=2)

# Small selected tab underline (structural accent)
# centered region under the middle-right area (do not draw text)
uline_w = 220
uline_h = 6
uline_x = (canvas.width // 2) - (uline_w // 2)
uline_y = tabs_top + 88
draw.rounded_rectangle([uline_x, uline_y, uline_x + uline_w, uline_y + uline_h], radius=4, fill=(60, 95, 220))

# Large content area background (placeholder panel behind posts/sections)
content_top = sep_y + 40
content_left = 40
content_right = canvas.width - 40
content_height = 900
draw.rectangle([content_left, content_top, content_right, content_top + content_height], fill=(250,250,252))

# Sub card areas inside content (two sample subtle cards)
card_pad = 24
inner_w = content_right - content_left - card_pad*2
card1_top = content_top + 24
card1_bottom = card1_top + 220
draw.rounded_rectangle([content_left + card_pad, card1_top, content_left + card_pad + inner_w, card1_bottom],
                       radius=18, fill=(245,245,247))

card2_top = card1_bottom + 36
card2_bottom = card2_top + 180
draw.rounded_rectangle([content_left + card_pad, card2_top, content_left + card_pad + inner_w, card2_bottom],
                       radius=18, fill=(245,245,247))

# Light separators to structure the lower page
for i in range(4):
    y = content_top + content_height - 40 - i*60
    draw.line([(content_left + 12, y), (content_right - 12, y)], fill=(240,240,242), width=1)

# Bottom edge fade to indicate scrollable content
fade_top = content_top + content_height - 80
for i in range(80):
    t = i / 79
    c = int(255 - t * 8)
    draw.line([(0, fade_top + i), (canvas.width, fade_top + i)], fill=(c, c, c))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/00_icon_Follow.png
try:
    _c0 = get_crop(0, 360, 132)
    canvas.paste(_c0, (540, 769), _c0)
except Exception:
    pass
layout["Follow"] = [540, 769, 900, 901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/01_icon_About.png
try:
    _c1 = get_crop(1, 215, 144)
    canvas.paste(_c1, (526, 953), _c1)
except Exception:
    pass
layout["About"] = [526, 953, 741, 1097]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/02_icon_Contact_organizer.png
try:
    _c2 = get_crop(2, 90, 90)
    canvas.paste(_c2, (1176, 180), _c2)
except Exception:
    pass
layout["Contact_organizer"] = [1176, 180, 1266, 270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/03_icon_Share_with_friends.png
try:
    _c3 = get_crop(3, 90, 90)
    canvas.paste(_c3, (1320, 180), _c3)
except Exception:
    pass
layout["Share_with_friends"] = [1320, 180, 1410, 270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/04_icon_Imy.png
try:
    _c4 = get_crop(4, 64, 67)
    canvas.paste(_c4, (179, 0), _c4)
except Exception:
    pass
layout["Imy"] = [179, 0, 243, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/05_icon_Imy.png
try:
    _c5 = get_crop(5, 67, 69)
    canvas.paste(_c5, (110, 0), _c5)
except Exception:
    pass
layout["Imy"] = [110, 0, 177, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/06_icon_5.21.png
try:
    _c6 = get_crop(6, 90, 90)
    canvas.paste(_c6, (36, 180), _c6)
except Exception:
    pass
layout["5.21"] = [36, 180, 126, 270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 69, 66)
    canvas.paste(_c7, (307, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [307, 0, 376, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/08_icon_Imy.png
try:
    _c8 = get_crop(8, 58, 67)
    canvas.paste(_c8, (245, 0), _c8)
except Exception:
    pass
layout["Imy"] = [245, 0, 303, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 67)
    canvas.paste(_c9, (382, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 0, 435, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 64)
    canvas.paste(_c10, (1316, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1316, 0, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 57, 63)
    canvas.paste(_c11, (1215, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1215, 1, 1272, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/12_icon_Quinton_Starks.png
try:
    _c12 = get_crop(12, 250, 245)
    canvas.paste(_c12, (592, 293), _c12)
except Exception:
    pass
layout["Quinton_Starks"] = [592, 293, 842, 538]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 42, 63)
    canvas.paste(_c13, (1272, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1272, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/14_icon_Collections.png
try:
    _c14 = get_crop(14, 316, 144)
    canvas.paste(_c14, (217, 949), _c14)
except Exception:
    pass
layout["Collections"] = [217, 949, 533, 1093]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/15_icon_5.21.png
try:
    _c15 = get_crop(15, 94, 63)
    canvas.paste(_c15, (13, 1), _c15)
except Exception:
    pass
layout["5.21"] = [13, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/16_icon_Report_this_profile.png
try:
    _c16 = get_crop(16, 1344, 65)
    canvas.paste(_c16, (48, 1363), _c16)
except Exception:
    pass
layout["Report_this_profile"] = [48, 1363, 1392, 1428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/17_text_8_16_followers.png
try:
    _c17 = get_crop(17, 360, 132)
    canvas.paste(_c17, (540, 769), _c17)
except Exception:
    pass
layout["8_16_followers"] = [540, 769, 900, 901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_07_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-9/18_text_Events.png
try:
    _c18 = get_crop(18, 217, 144)
    canvas.paste(_c18, (0, 949), _c18)
except Exception:
    pass
layout["Events"] = [0, 949, 217, 1093]
