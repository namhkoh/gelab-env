# page_id: page_eventbrite_92c22920a83749c994864397a370a984_13
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-15.png
# step_index: 13/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 canvas using PIL ImageDraw (canvas, draw provided)

# 1) Base background (slightly warm off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBF8FB")

# 2) Status bar area at top (~60px) - muted gray strip
status_h = 60
draw.rectangle([(0, 0), (1440, status_h)], fill="#BEBDBA")

# subtle thin divider under status bar
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill="#A7A5A3", width=1)

# 3) Top hero/banner background (dark red gradient area for event image)
hero_top = status_h
hero_bottom = 560
# simple vertical gradient from deep red to darker red
start_col = (130, 10, 10)   # dark red
end_col = (60, 8, 8)        # darker near bottom
steps = hero_bottom - hero_top
if steps < 1:
    steps = 1
for i in range(steps):
    r = int(start_col[0] + (end_col[0] - start_col[0]) * (i / steps))
    g = int(start_col[1] + (end_col[1] - start_col[1]) * (i / steps))
    b = int(start_col[2] + (end_col[2] - start_col[2]) * (i / steps))
    draw.line([(0, hero_top + i), (1440, hero_top + i)], fill=(r, g, b))

# subtle vignette: darken edges with translucent rectangles
edge_overlay_color = (0, 0, 0, 48)
# left and right edges
# draw semi-transparent by compositing onto canvas (create temp image)
# REMOVED: from PIL import Image
overlay = Image.new("RGBA", (1440, hero_bottom - hero_top), (0, 0, 0, 0))
ov_draw = Image.Image._new(overlay)  # placeholder to get object; we'll use ImageDraw below
# REMOVED: from PIL import ImageDraw as _IDraw
ovd = _IDraw.Draw(overlay)
ovd.rectangle([(0, 0), (140, hero_bottom - hero_top)], fill=(0, 0, 0, 30))
ovd.rectangle([(1300, 0), (1440, hero_bottom - hero_top)], fill=(0, 0, 0, 30))
# paste overlay
canvas.paste(overlay, (0, hero_top), overlay)

# 4) Large subtle white content area under hero starts around y=560 (main content)
content_top = hero_bottom
draw.rectangle([(0, content_top), (1440, 2960)], fill="#FBF8FB")

# 5) Organizer card (rounded rectangle) around the organizer area
org_box = (48, 1100, 1392, 1250)
draw.rounded_rectangle(org_box, radius=28, fill="#FFFFFF", outline="#EDE8EE", width=1)

# 6) Light divider line under the organizer/card area
draw.line([(48, org_box[3] + 24), (1392, org_box[3] + 24)], fill="#E8E6EA", width=1)

# 7) Horizontal separators between major sections
# separator under the refund/policy area (approx)
sep_y1 = 1720
draw.line([(48, sep_y1), (1392, sep_y1)], fill="#ECE9ED", width=1)

# separator under event description area (approx)
sep_y2 = 2180
draw.line([(48, sep_y2), (1392, sep_y2)], fill="#F1EFF2", width=1)

# 8) "About this event" card background hint (slightly different white block)
about_box = (48, 1820, 1392, 2120)
draw.rectangle(about_box, fill="#FBF8FB", outline=None)

# 9) Location section background strip (subtle)
loc_top = 2520
loc_h = 220
draw.rectangle([(0, loc_top), (1440, loc_top + loc_h)], fill="#FFFFFF")
# top divider for location area
draw.line([(48, loc_top), (1392, loc_top)], fill="#ECE9ED", width=1)

# 10) Footer ticket bar background (pale lavender strip)
footer_top = 2720
footer_bottom = 2960
draw.rectangle([(0, footer_top), (1440, footer_bottom)], fill="#F6F4F8")
# top border line for footer
draw.line([(0, footer_top), (1440, footer_top)], fill="#E7E4E8", width=1)

# 11) Price area left side: leave clear area but give subtle grouping with a faint rounded rect background
price_capsule = (32, footer_top + 16, 480, footer_bottom - 16)
draw.rounded_rectangle(price_capsule, radius=12, fill="#F6F4F8", outline=None)

# 12) Ensure not to draw or duplicate any detected icon or text areas:
# (No text/icons drawn here; only background shapes and separators.)

# End of drawing structure.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/02_icon_May_M.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["May_M"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/03_icon_May_M.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["May_M"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/04_icon_5.01.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["5.01"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 47, 65)
    canvas.paste(_c5, (1156, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [1156, 2, 1203, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/06_icon_Sports_Fitness_._Wrestling.png
try:
    _c6 = get_crop(6, 234, 144)
    canvas.paste(_c6, (48, 2332), _c6)
except Exception:
    pass
layout["Sports_&_Fitness_._Wrestl"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/07_icon_Few_tickets_left.png
try:
    _c7 = get_crop(7, 430, 85)
    canvas.paste(_c7, (41, 753), _c7)
except Exception:
    pass
layout["Few_tickets_left"] = [41, 753, 471, 838]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 44, 62)
    canvas.paste(_c8, (1328, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1328, 3, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/09_icon_Imy.png
try:
    _c9 = get_crop(9, 62, 69)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["Imy"] = [180, 0, 242, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/10_icon_Imy.png
try:
    _c10 = get_crop(10, 60, 70)
    canvas.paste(_c10, (115, 0), _c10)
except Exception:
    pass
layout["Imy"] = [115, 0, 175, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/11_icon_Show_map.png
try:
    _c11 = get_crop(11, 226, 144)
    canvas.paste(_c11, (1166, 2550), _c11)
except Exception:
    pass
layout["Show_map"] = [1166, 2550, 1392, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/12_icon_1.Sk_Followers.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (96, 1194), _c12)
except Exception:
    pass
layout["1.Sk_Followers"] = [96, 1194, 240, 1338]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/13_icon_Imy.png
try:
    _c13 = get_crop(13, 56, 67)
    canvas.paste(_c13, (246, 1), _c13)
except Exception:
    pass
layout["Imy"] = [246, 1, 302, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 100, 62)
    canvas.paste(_c14, (1214, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 2, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/15_icon_Major_League_Wrestling_presents_MLW_AZTE.png
try:
    _c15 = get_crop(15, 234, 144)
    canvas.paste(_c15, (48, 2332), _c15)
except Exception:
    pass
layout["Major_League_Wrestling_pr"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 64, 67)
    canvas.paste(_c16, (310, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [310, 1, 374, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/17_text_5.01.png
try:
    _c17 = get_crop(17, 87, 43)
    canvas.paste(_c17, (22, 17), _c17)
except Exception:
    pass
layout["5.01"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/18_text_ZTEC.png
try:
    _c18 = get_crop(18, 146, 55)
    canvas.paste(_c18, (751, 148), _c18)
except Exception:
    pass
layout["ZTEC"] = [751, 148, 897, 203]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/19_text_Fjumtg.png
try:
    _c19 = get_crop(19, 67, 33)
    canvas.paste(_c19, (787, 297), _c19)
except Exception:
    pass
layout["Fjumtg"] = [787, 297, 854, 330]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/20_text_StURDaY.png
try:
    _c20 = get_crop(20, 113, 18)
    canvas.paste(_c20, (924, 334), _c20)
except Exception:
    pass
layout["StURDaY"] = [924, 334, 1037, 352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/21_text_May_M.png
try:
    _c21 = get_crop(21, 74, 18)
    canvas.paste(_c21, (1046, 334), _c21)
except Exception:
    pass
layout["May_M"] = [1046, 334, 1120, 352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/22_text_Tore.png
try:
    _c22 = get_crop(22, 59, 14)
    canvas.paste(_c22, (924, 380), _c22)
except Exception:
    pass
layout["Tore"] = [924, 380, 983, 394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/23_text_M.png
try:
    _c23 = get_crop(23, 32, 9)
    canvas.paste(_c23, (1047, 381), _c23)
except Exception:
    pass
layout["M_"] = [1047, 381, 1079, 390]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/24_text_Saturday_May_11.png
try:
    _c24 = get_crop(24, 435, 74)
    canvas.paste(_c24, (38, 887), _c24)
except Exception:
    pass
layout["Saturday;_May_11"] = [38, 887, 473, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/25_text_7_00_PM.png
try:
    _c25 = get_crop(25, 209, 54)
    canvas.paste(_c25, (511, 895), _c25)
except Exception:
    pass
layout["7:00_PM"] = [511, 895, 720, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/26_text_MLW_AZTECA_LUCHA_Triller_TV_PPV.png
try:
    _c26 = get_crop(26, 509, 144)
    canvas.paste(_c26, (288, 1155), _c26)
except Exception:
    pass
layout["MLW:_AZTECA_LUCHA_(Trille"] = [288, 1155, 797, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/27_text_Major_League_Wrestling.png
try:
    _c27 = get_crop(27, 509, 144)
    canvas.paste(_c27, (288, 1155), _c27)
except Exception:
    pass
layout["Major_League_Wrestling"] = [288, 1155, 797, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/28_text_1.Sk_Followers.png
try:
    _c28 = get_crop(28, 509, 144)
    canvas.paste(_c28, (288, 1155), _c28)
except Exception:
    pass
layout["1.Sk_Followers"] = [288, 1155, 797, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/29_text_Cicero_Stadium.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1422), _c29)
except Exception:
    pass
layout["Cicero_Stadium"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/30_text_4hrs.png
try:
    _c30 = get_crop(30, 112, 49)
    canvas.paste(_c30, (141, 1580), _c30)
except Exception:
    pass
layout["4hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/31_text_Refund_policy.png
try:
    _c31 = get_crop(31, 299, 63)
    canvas.paste(_c31, (138, 1685), _c31)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/32_text_The_organizer_will_review_refund_request.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1422), _c32)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/33_text_Location.png
try:
    _c33 = get_crop(33, 246, 63)
    canvas.paste(_c33, (41, 2594), _c33)
except Exception:
    pass
layout["Location"] = [41, 2594, 287, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_13_2024_4_24_16_59_92c22920a83749c994864397a370a984-15/34_text_S10_-_95.png
try:
    _c34 = get_crop(34, 225, 61)
    canvas.paste(_c34, (89, 2811), _c34)
except Exception:
    pass
layout["S10_-_$95"] = [89, 2811, 314, 2872]
