# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_08
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10.png
# step_index: 8/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the calendar/date-picker page.
# Available variables: canvas (PIL Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors (matching screenshot's subtle purples / grays)
bg_white = (255, 255, 255)
status_gray = (199, 199, 201)        # status bar background
muted_divider = (225, 219, 235)      # faint purple/gray divider
card_border = (230, 224, 239)        # very light purple border for cards
soft_purple_bg = (250, 248, 252)     # extremely light purple tint for large sections
deep_divider = (210, 204, 224)       # slightly stronger divider

# Fill overall background (canvas may already be white, but ensure consistent)
draw.rectangle([(0,0),(w,h)], fill=bg_white)

# Status bar area (top strip for time/signal icons)
status_h = 72
draw.rectangle([(0,0),(w,status_h)], fill=status_gray)

# Header / toolbar area under status bar
header_top = status_h
header_h = 120
header_bottom = header_top + header_h
# keep header background same as canvas (white) but draw subtle bottom divider/shadow
draw.rectangle([(0, header_top),(w, header_bottom)], fill=bg_white)
# subtle divider line under header
draw.line([(40, header_bottom-1),(w-40, header_bottom-1)], fill=muted_divider, width=2)

# Decorative small underline centered for the title area (doesn't duplicate text)
title_line_y = header_top + header_h//2 + 18
draw.line([(w*0.35, title_line_y),(w*0.65, title_line_y)], fill=muted_divider, width=3)

# Calendar card background (rounded rectangle) that holds the month and calendar grid
cal_x0, cal_y0 = 40, header_bottom + 40
cal_x1, cal_y1 = w - 40, cal_y0 + 960
cal_radius = 20
# white fill so calendar numbers/icons pasted on top remain visible; border to define section
draw.rounded_rectangle([(cal_x0, cal_y0),(cal_x1, cal_y1)], radius=cal_radius,
                       fill=bg_white, outline=card_border, width=2)

# Add very subtle vertical center guide (purely structural, very faint)
draw.line([((w//2), cal_y0+20), ((w//2), cal_y1-20)], fill=deep_divider, width=1)

# Month navigation row separator (thin line under month heading)
month_sep_y = cal_y0 + 160
draw.line([(cal_x0+24, month_sep_y),(cal_x1-24, month_sep_y)], fill=muted_divider, width=1)

# Calendar weekday labels separator (subtle)
weekday_sep_y = month_sep_y + 70
draw.line([(cal_x0+24, weekday_sep_y),(cal_x1-24, weekday_sep_y)], fill=muted_divider, width=1)

# Highlight area behind the grid (very subtle)
grid_bg_y0 = weekday_sep_y + 10
grid_bg_y1 = cal_y1 - 70
draw.rectangle([(cal_x0+16, grid_bg_y0),(cal_x1-16, grid_bg_y1)], fill=soft_purple_bg)

# End Date section card (below calendar) - very light tinted background to group end-date area
end_x0, end_y0 = 40, cal_y1 + 60
end_x1, end_y1 = w - 40, end_y0 + 360
draw.rounded_rectangle([(end_x0, end_y0),(end_x1, end_y1)], radius=18,
                       fill=bg_white, outline=card_border, width=1)

# Very light inner panel behind "Choose a date" area to give depth (don't draw text)
inner_panel_margin = 18
draw.rectangle([(end_x0+inner_panel_margin, end_y0+inner_panel_margin),
                (end_x1-inner_panel_margin, end_y0+120)], fill=soft_purple_bg, outline=None)

# Large empty content area remains white below end-date section (no extra drawing)

# Separator line above the bottom action bar (just above Apply button area)
apply_bar_top = h - 200
draw.line([(24, apply_bar_top),(w-24, apply_bar_top)], fill=muted_divider, width=2)
# Slight shadow above apply area
for i, alpha in enumerate([10, 8, 6]):
    draw.line([(24, apply_bar_top+1+i),(w-24, apply_bar_top+1+i)],
              fill=(200, 196, 213), width=1)

# Add subtle rounded border where the Apply button will appear (light outline only,
# so the actual button graphic pasted later won't be duplicated)
apply_box = [(48, h-200+24),(w-48, h-56)]
draw.rounded_rectangle(apply_box, radius=12, outline=card_border, width=3, fill=None)

# Final subtle accents: small left margin vertical guide and right margin guide (for layout alignment)
draw.line([(40, header_bottom+10),(40, h-220)], fill=deep_divider, width=1)
draw.line([(w-40, header_bottom+10),(w-40, h-220)], fill=deep_divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/00_icon_28.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (60, 1201), _c0)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/02_icon_29.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (192, 1201), _c2)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/04_icon_30.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1201), _c4)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/05_icon_23.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (324, 1081), _c5)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/06_icon_5.23.png
try:
    _c6 = get_crop(6, 62, 66)
    canvas.paste(_c6, (179, 1), _c6)
except Exception:
    pass
layout["5.23"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/07_icon_5.23.png
try:
    _c7 = get_crop(7, 62, 66)
    canvas.paste(_c7, (113, 1), _c7)
except Exception:
    pass
layout["5.23"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/08_icon_25.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (588, 1081), _c8)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 63, 64)
    canvas.paste(_c9, (309, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [309, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/10_icon_22.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (192, 1081), _c10)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 64)
    canvas.paste(_c11, (247, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 69)
    canvas.paste(_c12, (1316, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 0, 1373, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 90, 69)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1302, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/14_icon_5.23.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (12, 72), _c14)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/15_icon_26.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (720, 1081), _c15)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1081), _c16)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/17_icon_5.23.png
try:
    _c17 = get_crop(17, 92, 64)
    canvas.paste(_c17, (16, 1), _c17)
except Exception:
    pass
layout["5.23"] = [16, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 49, 67)
    canvas.paste(_c18, (382, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 41, 65)
    canvas.paste(_c19, (1274, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/20_icon_11.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (588, 721), _c20)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/21_icon_Choose_a_date.png
try:
    _c21 = get_crop(21, 638, 144)
    canvas.paste(_c21, (48, 1490), _c21)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/22_icon_Next_month.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (846, 457), _c22)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/23_icon_What_date.png
try:
    _c23 = get_crop(23, 322, 71)
    canvas.paste(_c23, (558, 113), _c23)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/24_icon_April_2024.png
try:
    _c24 = get_crop(24, 121, 110)
    canvas.paste(_c24, (596, 611), _c24)
except Exception:
    pass
layout["April_2024"] = [596, 611, 717, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 583, 114)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/26_text_April_2024.png
try:
    _c26 = get_crop(26, 203, 54)
    canvas.paste(_c26, (420, 504), _c26)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/27_text_10.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 841), _c27)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/28_text_11.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 841), _c28)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/29_text_12.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 841), _c29)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/30_text_13.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 841), _c30)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/31_text_14.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 961), _c31)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/32_text_15.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 961), _c32)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/33_text_16.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 961), _c33)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/34_text_17.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 961), _c34)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/35_text_18.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 961), _c35)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/36_text_19.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 961), _c36)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/37_text_20.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (852, 961), _c37)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/38_text_21.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (60, 1081), _c38)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/39_clickable_1.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (192, 721), _c39)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/40_clickable_2.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (324, 721), _c40)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/41_clickable_3.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 721), _c41)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/42_clickable_5.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 721), _c42)
except Exception:
    pass
layout["5"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/43_clickable_6.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 721), _c43)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/44_clickable_7.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 841), _c44)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/45_clickable_8.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 841), _c45)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_08_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-10/46_clickable_9.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 841), _c46)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
