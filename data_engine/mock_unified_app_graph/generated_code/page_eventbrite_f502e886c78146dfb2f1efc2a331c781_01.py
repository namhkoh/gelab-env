# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_01
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3.png
# step_index: 1/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas.
# Available variables: canvas (PIL Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# 1) Base background (ensure clean white)
draw.rectangle([(0, 0), (w, h)], fill="#FFFFFF")

# 2) Status bar area at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill="#CFCFCF")

# subtle separator under status bar
draw.line([(0, status_h), (w, status_h)], fill="#BDBDBD", width=1)

# 3) Top toolbar area (behind search bar and logo)
toolbar_top = status_h
toolbar_bottom = 200
draw.rectangle([(0, toolbar_top), (w, toolbar_bottom)], fill="#FFFFFF")
# toolbar bottom divider
draw.line([(48, toolbar_bottom), (w-48, toolbar_bottom)], fill="#EEEAF2", width=2)

# 4) Main content column background (the central column that holds event rows)
# This sits with left margin 48 and right margin 48 as in screenshot detection.
col_left = 48
col_right = w - 48
col_top = 420
col_bottom = 2600
draw.rounded_rectangle([(col_left, col_top), (col_right, col_bottom)],
                       radius=20, fill="#FBFBFD", outline="#F0EDF6", width=1)

# 5) Row separators between event items (thin lines)
# Using detected row heights: many items are 396px tall, so separators at these y positions:
separator_ys = [886, 1282, 1678, 2074, 2470]
sep_x0 = col_left + 12
sep_x1 = col_right - 12
for y in separator_ys:
    # subtle divider
    draw.line([(sep_x0, y), (sep_x1, y)], fill="#E8E6EB", width=1)

# 6) Soft inner shadow at top of content column to separate header from list
# Draw a faint horizontal gradient-like band using semi-opaque lines (approximation)
grad_top = col_top
grad_height = 10
for i in range(grad_height):
    alpha = int(12 * (1 - i / grad_height))  # fade out
    # convert to RGBA blended over white background by computing blended color manually
    base = 0xF7  # near-white band color base
    band_color = (240, 238, 246)  # light lilac tone
    # slightly darken with alpha effect on white
    draw.line([(col_left + 6, grad_top + i), (col_right - 6, grad_top + i)],
              fill=band_color, width=1)

# 7) Accent left gutter (very subtle) to visually anchor list
gutter_w = 6
draw.rectangle([(col_left + 2, col_top + 12), (col_left + 2 + gutter_w, col_bottom - 12)], fill="#F2EFFF")

# 8) Bottom navigation bar background (slot for tabs)
nav_top = 2804
nav_bottom = h
draw.rectangle([(0, nav_top), (w, nav_bottom)], fill="#FFFFFF")
# nav top divider and slight shadow
draw.line([(0, nav_top), (w, nav_top)], fill="#E6E6E6", width=1)
# subtle drop shadow under nav (a few semi-opaque lines)
shadow_h = 6
for i in range(shadow_h):
    shade = 220 - i * 10
    if shade < 0: shade = 0
    draw.line([(0, nav_top + i), (w, nav_top + i)], fill=(shade, shade, shade), width=1)

# 9) Small floating card background placeholder area (only the background shape, not content)
# There's a floating location pill in the screenshot which will be pasted; draw a soft backdrop behind where it sits
# But keep it very faint so pasted element sits clearly on top. Position approximate but intentionally faint.
pill_box = (420, 2590, 980, 2670)  # approximate container behind location pill; kept very subtle
draw.rounded_rectangle([pill_box[0], pill_box[1], pill_box[2], pill_box[3]],
                       radius=28, fill="#FFFFFF", outline="#EDEAF1", width=1)

# 10) Provide subtle page edge guides (left/right) to match UI margins
# Thin vertical guides (very faint) to mark safe content area (for visual structure only)
draw.line([(col_left, toolbar_bottom + 6), (col_left, nav_top - 6)], fill="#F5F4F7", width=1)
draw.line([(col_right, toolbar_bottom + 6), (col_right, nav_top - 6)], fill="#F5F4F7", width=1)

# 11) Final subtle vignette around content column to anchor it on the white canvas (very light)
# Draw a faint outline
draw.rounded_rectangle([(col_left-2, col_top-2), (col_right+2, col_bottom+2)],
                       radius=22, outline="#F3F1F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/00_icon_Ibaigktsinel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1282), _c0)
except Exception:
    pass
layout["Ibaigktsinel"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/01_icon_TAKE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["'TAKE"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/02_icon_NDIE_DANCEPA.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/03_icon_Q_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 456, 117)
    canvas.paste(_c4, (492, 2651), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/05_icon_NDIE.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["NDIE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1555), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1284, 1555), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/11_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 490), _c11)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/12_icon_Sylmai.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 2347), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1143), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/15_icon_Club_Decades.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1140, 1143), _c15)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1140, 763), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/17_icon_Home.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (0, 2804), _c17)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 60)
    canvas.paste(_c18, (311, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [311, 2, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1284, 763), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/20_icon_7.18.png
try:
    _c20 = get_crop(20, 57, 61)
    canvas.paste(_c20, (182, 2), _c20)
except Exception:
    pass
layout["7.18"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/21_icon_59_creator_followers.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1678), _c21)
except Exception:
    pass
layout["59_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 52, 60)
    canvas.paste(_c22, (247, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/23_icon_7.18.png
try:
    _c23 = get_crop(23, 100, 97)
    canvas.paste(_c23, (42, 123), _c23)
except Exception:
    pass
layout["7.18"] = [42, 123, 142, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 48, 53)
    canvas.paste(_c24, (1320, 7), _c24)
except Exception:
    pass
layout["icon_24"] = [1320, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/25_icon_8_4717_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 886), _c25)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 88, 58)
    canvas.paste(_c26, (1212, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1212, 4, 1300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/27_icon_Free.png
try:
    _c27 = get_crop(27, 1344, 346)
    canvas.paste(_c27, (48, 2470), _c27)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/28_icon_7.18.png
try:
    _c28 = get_crop(28, 58, 62)
    canvas.paste(_c28, (115, 1), _c28)
except Exception:
    pass
layout["7.18"] = [115, 1, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/29_icon_Public_House_Los_Angeles_CA.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 490), _c29)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/30_icon_Q_Search_events.png
try:
    _c30 = get_crop(30, 44, 57)
    canvas.paste(_c30, (385, 6), _c30)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/31_icon_8_21119_creator_followers.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1282), _c31)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/32_icon_9.30_PM_PDT.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["9.30_PM_PDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/33_icon_The_Grand_Star_Jazz_Club.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["The_Grand_Star_Jazz_Club"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/34_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/35_icon_icon_35.png
try:
    _c35 = get_crop(35, 41, 56)
    canvas.paste(_c35, (1272, 5), _c35)
except Exception:
    pass
layout["icon_35"] = [1272, 5, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/36_icon_Free.png
try:
    _c36 = get_crop(36, 127, 73)
    canvas.paste(_c36, (247, 1749), _c36)
except Exception:
    pass
layout["Free"] = [247, 1749, 374, 1822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/37_icon_8_21119_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1282), _c37)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/38_text_7.18.png
try:
    _c38 = get_crop(38, 92, 41)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["7.18"] = [22, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/40_text_Mon_May_13.png
try:
    _c40 = get_crop(40, 222, 43)
    canvas.paste(_c40, (393, 2525), _c40)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/41_text_5.30_PM_PDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/42_text_Grief_Loss_Resiliency.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["Grief;_Loss,_Resiliency"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/43_text_Afliccion_Perdida_y.png
try:
    _c43 = get_crop(43, 144, 123)
    canvas.paste(_c43, (1140, 2347), _c43)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/44_text_31_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/45_clickable_Favorites.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (576, 2804), _c45)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/46_clickable_Tickets.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (864, 2804), _c46)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_01_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-3/47_clickable_More.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (1152, 2804), _c47)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
