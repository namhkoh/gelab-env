# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_06
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8.png
# step_index: 6/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar, header banner, cards and separators for the mobile UI layout
w, h = canvas.size

# Colors
status_bar_color = (189, 189, 189)      # light gray status bar
banner_left = (166, 131, 116)           # warm brown left of banner
banner_right = (232, 226, 222)          # pale neutral right of banner
overlay_circle = (255, 255, 255, 20)    # subtle highlight (used via alpha simulation)
card_shadow = (220, 220, 220)           # shadow for card
card_fill = (255, 255, 255)             # white card
divider = (235, 235, 235)               # light divider
tab_indicator = (66, 103, 255)          # blue tab underline
section_bg = (250, 250, 250)            # very light section background
row_sep = (245, 245, 245)               # row separator

# Dimensions (proportional to canvas)
status_h = int(56 * (w / 1440))         # keep roughly 56 px scaled horizontally
banner_h = int(360 * (w / 1440))
banner_top = status_h
banner_bottom = banner_top + banner_h

# Draw status bar (solid)
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Draw horizontal gradient banner (left->right)
for x in range(w):
    # interpolate between banner_left and banner_right
    t = x / (w - 1)
    r = int(banner_left[0] * (1 - t) + banner_right[0] * t)
    g = int(banner_left[1] * (1 - t) + banner_right[1] * t)
    b = int(banner_left[2] * (1 - t) + banner_right[2] * t)
    draw.line([(x, banner_top), (x, banner_bottom)], fill=(r, g, b))

# Subtle radial-ish highlight (simulate blurred center)
# Draw several translucent concentric ellipses to create a soft glow
center_x = w // 2
center_y = banner_top + int(banner_h * 0.35)
max_rad_x = int(w * 0.6)
max_rad_y = int(banner_h * 0.9)
# draw decreasing alpha ellipses by stepping
for i, alpha in enumerate([24, 18, 12, 8, 4]):
    rx = int(max_rad_x * (1 - i * 0.15))
    ry = int(max_rad_y * (1 - i * 0.15))
    bbox = [center_x - rx, center_y - ry, center_x + rx, center_y + ry]
    # simulate alpha by blending onto canvas via a temporary fill of slightly different color
    draw.ellipse(bbox, fill=(255, 255, 255, 0))

# Draw a large white rounded card overlapping the banner to host profile/name area
card_margin = int(w * 0.03)
card_left = card_margin
card_right = w - card_margin
card_top = int(banner_bottom - (banner_h * 0.25))
card_bottom = card_top + int(h * 0.28)
card_radius = 28

# Shadow: draw a slightly offset rounded rectangle as shadow
shadow_offset = int(12 * (w / 1440))
shadow_box = [card_left + shadow_offset, card_top + shadow_offset,
              card_right + shadow_offset, card_bottom + shadow_offset]
draw.rounded_rectangle(shadow_box, radius=card_radius, fill=card_shadow)

# Card itself
draw.rounded_rectangle([card_left, card_top, card_right, card_bottom],
                       radius=card_radius, fill=card_fill)

# Tabs area: draw a light separator line below the card where tabs live
tabs_y = card_bottom + int(h * 0.02)
draw.line([(card_left + 8, tabs_y), (card_right - 8, tabs_y)], fill=divider, width=2)

# Tabs background band (slightly off-white)
tabs_band_h = int(110 * (w / 1440))
tabs_band_top = tabs_y
tabs_band_bottom = tabs_band_top + tabs_band_h
draw.rectangle([(0, tabs_band_top), (w, tabs_band_bottom)], fill=section_bg)

# Underline indicator for the active tab (left side)
indicator_w = int(w * 0.12)
indicator_h = int(6 * (w / 1440))
indicator_x = card_left + int(20 * (w / 1440))
indicator_y = tabs_band_bottom - indicator_h - int(12 * (w / 1440))
draw.rectangle([(indicator_x, indicator_y), (indicator_x + indicator_w, indicator_y + indicator_h)],
               fill=tab_indicator)

# Main content divider under tabs
content_top = tabs_band_bottom + int(h * 0.02)
draw.line([(card_left + 8, content_top), (card_right - 8, content_top)], fill=divider, width=1)

# "Upcoming" section title space background (keeps consistency, subtle)
upcoming_pad_top = content_top + int(h * 0.02)
upcoming_pad_bottom = upcoming_pad_top + int(h * 0.08)
# Keep it white but draw a slight left margin accent
draw.rectangle([(card_left + 8, upcoming_pad_top), (card_right - 8, upcoming_pad_bottom)], fill=card_fill)

# Draw separators for a couple of list rows (visual structure for event list)
row_height = int(220 * (w / 1440))
first_row_top = upcoming_pad_bottom + int(h * 0.02)
second_row_top = first_row_top + row_height + int(h * 0.04)

# Left thumbnail placeholders as soft rounded rectangles (these will be replaced by pasted thumbnails)
thumb_w = int(260 * (w / 1440))
thumb_h = int(160 * (w / 1440))
thumb_radius = 12
thumb_x = card_left + int(16 * (w / 1440))
thumb_y1 = first_row_top
thumb_y2 = second_row_top

# Light placeholder backgrounds (very faint)
draw.rounded_rectangle([thumb_x, thumb_y1, thumb_x + thumb_w, thumb_y1 + thumb_h],
                       radius=thumb_radius, fill=section_bg, outline=divider)
draw.rounded_rectangle([thumb_x, thumb_y2, thumb_x + thumb_w, thumb_y2 + thumb_h],
                       radius=thumb_radius, fill=section_bg, outline=divider)

# Row separators (horizontal)
sep_x1 = thumb_x + thumb_w + int(24 * (w / 1440))
sep_x2 = card_right - int(24 * (w / 1440))
draw.line([(card_left + 8, first_row_top + thumb_h + int(18 * (w / 1440))),
           (card_right - 8, first_row_top + thumb_h + int(18 * (w / 1440)))],
          fill=row_sep, width=2)
draw.line([(card_left + 8, second_row_top + thumb_h + int(18 * (w / 1440))),
           (card_right - 8, second_row_top + thumb_h + int(18 * (w / 1440)))],
          fill=row_sep, width=2)

# Add subtle divider lines where share/like icons appear on the right to guide layout (very faint)
right_icon_guides_x = card_right - int(120 * (w / 1440))
draw.line([(right_icon_guides_x, first_row_top + int(10 * (w / 1440))),
           (right_icon_guides_x, first_row_top + thumb_h - int(10 * (w / 1440)))],
          fill=(245, 245, 245), width=1)
draw.line([(right_icon_guides_x, second_row_top + int(10 * (w / 1440))),
           (right_icon_guides_x, second_row_top + thumb_h - int(10 * (w / 1440)))],
          fill=(245, 245, 245), width=1)

# Footer subtle gradient to keep bottom visually balanced
footer_h = int(h * 0.06)
for i in range(footer_h):
    t = i / max(1, footer_h - 1)
    gray = int(255 - t * 6)
    draw.line([(0, h - footer_h + i), (w, h - footer_h + i)], fill=(gray, gray, gray))

# Done - structural elements drawn. Texts/icons will be pasted on top externally.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/00_icon_Follow.png
try:
    _c0 = get_crop(0, 360, 132)
    canvas.paste(_c0, (540, 769), _c0)
except Exception:
    pass
layout["Follow"] = [540, 769, 900, 901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/01_icon_caoui.png
try:
    _c1 = get_crop(1, 1440, 396)
    canvas.paste(_c1, (0, 1294), _c1)
except Exception:
    pass
layout["caoui"] = [0, 1294, 1440, 1690]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/02_icon_Contact_organizer.png
try:
    _c2 = get_crop(2, 90, 90)
    canvas.paste(_c2, (1176, 180), _c2)
except Exception:
    pass
layout["Contact_organizer"] = [1176, 180, 1266, 270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/03_icon_Share_with_friends.png
try:
    _c3 = get_crop(3, 90, 90)
    canvas.paste(_c3, (1320, 180), _c3)
except Exception:
    pass
layout["Share_with_friends"] = [1320, 180, 1410, 270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/04_icon_Events.png
try:
    _c4 = get_crop(4, 221, 145)
    canvas.paste(_c4, (0, 951), _c4)
except Exception:
    pass
layout["Events"] = [0, 951, 221, 1096]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/05_icon_San_Antonio_Texas_Property_Tax.png
try:
    _c5 = get_crop(5, 1440, 396)
    canvas.paste(_c5, (0, 1294), _c5)
except Exception:
    pass
layout["San_Antonio_Texas_Propert"] = [0, 1294, 1440, 1690]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/06_icon_Imy.png
try:
    _c6 = get_crop(6, 66, 70)
    canvas.paste(_c6, (110, 0), _c6)
except Exception:
    pass
layout["Imy"] = [110, 0, 176, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/07_icon_Imy.png
try:
    _c7 = get_crop(7, 63, 68)
    canvas.paste(_c7, (179, 0), _c7)
except Exception:
    pass
layout["Imy"] = [179, 0, 242, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 68, 68)
    canvas.paste(_c8, (307, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [307, 0, 375, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/09_icon_Imy.png
try:
    _c9 = get_crop(9, 56, 68)
    canvas.paste(_c9, (246, 0), _c9)
except Exception:
    pass
layout["Imy"] = [246, 0, 302, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/10_icon_5.21.png
try:
    _c10 = get_crop(10, 90, 90)
    canvas.paste(_c10, (36, 180), _c10)
except Exception:
    pass
layout["5.21"] = [36, 180, 126, 270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/11_icon_Collections.png
try:
    _c11 = get_crop(11, 316, 144)
    canvas.paste(_c11, (217, 949), _c11)
except Exception:
    pass
layout["Collections"] = [217, 949, 533, 1093]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/12_icon_Quinton_Starks.png
try:
    _c12 = get_crop(12, 251, 241)
    canvas.paste(_c12, (591, 293), _c12)
except Exception:
    pass
layout["Quinton_Starks"] = [591, 293, 842, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/13_icon_Quinton_Starks.png
try:
    _c13 = get_crop(13, 360, 132)
    canvas.paste(_c13, (540, 769), _c13)
except Exception:
    pass
layout["Quinton_Starks"] = [540, 769, 900, 901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/14_icon_Like_event.png
try:
    _c14 = get_crop(14, 72, 72)
    canvas.paste(_c14, (1320, 1966), _c14)
except Exception:
    pass
layout["Like_event"] = [1320, 1966, 1392, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 55, 64)
    canvas.paste(_c15, (1317, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1317, 0, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 68)
    canvas.paste(_c16, (382, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 0, 434, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 41, 63)
    canvas.paste(_c17, (1273, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1273, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 58, 60)
    canvas.paste(_c18, (1214, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1214, 3, 1272, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/19_icon_Like_event.png
try:
    _c19 = get_crop(19, 72, 72)
    canvas.paste(_c19, (1320, 1570), _c19)
except Exception:
    pass
layout["Like_event"] = [1320, 1570, 1392, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/20_icon_Share_Event.png
try:
    _c20 = get_crop(20, 72, 72)
    canvas.paste(_c20, (1200, 1966), _c20)
except Exception:
    pass
layout["Share_Event"] = [1200, 1966, 1272, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/21_icon_About.png
try:
    _c21 = get_crop(21, 202, 144)
    canvas.paste(_c21, (533, 949), _c21)
except Exception:
    pass
layout["About"] = [533, 949, 735, 1093]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/22_icon_Share_Event.png
try:
    _c22 = get_crop(22, 72, 72)
    canvas.paste(_c22, (1200, 1570), _c22)
except Exception:
    pass
layout["Share_Event"] = [1200, 1570, 1272, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/23_icon_San_Antonio_Texas_Property_Tax.png
try:
    _c23 = get_crop(23, 1440, 396)
    canvas.paste(_c23, (0, 1294), _c23)
except Exception:
    pass
layout["San_Antonio_Texas_Propert"] = [0, 1294, 1440, 1690]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/24_icon_Itobon.png
try:
    _c24 = get_crop(24, 92, 135)
    canvas.paste(_c24, (31, 1898), _c24)
except Exception:
    pass
layout["Itobon"] = [31, 1898, 123, 2033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/25_icon_11.png
try:
    _c25 = get_crop(25, 1440, 396)
    canvas.paste(_c25, (0, 1690), _c25)
except Exception:
    pass
layout["11"] = [0, 1690, 1440, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/26_icon_5.21.png
try:
    _c26 = get_crop(26, 100, 68)
    canvas.paste(_c26, (9, 0), _c26)
except Exception:
    pass
layout["5.21"] = [9, 0, 109, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/27_icon_ZOOM_Texas_Property_Tax_Protest_Seminar.png
try:
    _c27 = get_crop(27, 1440, 396)
    canvas.paste(_c27, (0, 1690), _c27)
except Exception:
    pass
layout["ZOOM_Texas_Property_Tax_P"] = [0, 1690, 1440, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/28_text_8_16_followers.png
try:
    _c28 = get_crop(28, 360, 132)
    canvas.paste(_c28, (540, 769), _c28)
except Exception:
    pass
layout["8_16_followers"] = [540, 769, 900, 901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_06_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-8/29_text_Upcoming.png
try:
    _c29 = get_crop(29, 290, 74)
    canvas.paste(_c29, (44, 1203), _c29)
except Exception:
    pass
layout["Upcoming"] = [44, 1203, 334, 1277]
