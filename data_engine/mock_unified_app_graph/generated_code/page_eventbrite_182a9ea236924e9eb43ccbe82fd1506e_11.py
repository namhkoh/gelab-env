# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_11
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13.png
# step_index: 11/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page
# Assumes the following variables are available:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object associated with canvas
# - font_sm, font_md, font_lg, font_xl (unused here)

W, H = canvas.size

# 1) Fill overall background (slightly warm white to match screenshot)
draw.rectangle([(0, 0), (W, H)], fill="#FFFFFF")

# 2) Status bar area at top (~56px)
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill="#D1D3D6")  # light grey status bar

# 3) Top header / hero banner with a vertical gradient (teal -> deep blue)
banner_top = status_h
banner_bottom = 520
steps = banner_bottom - banner_top
start_color = (16, 99, 115)   # deep teal
end_color   = (117, 173, 190) # lighter teal
for i in range(steps):
    t = i / max(steps - 1, 1)
    r = int(start_color[0] * (1 - t) + end_color[0] * t)
    g = int(start_color[1] * (1 - t) + end_color[1] * t)
    b = int(start_color[2] * (1 - t) + end_color[2] * t)
    draw.line([(0, banner_top + i), (W, banner_top + i)], fill=(r, g, b))

# Subtle darker bottom strip on banner to help overlaid title contrast
overlay_height = 120
overlay_color = (12, 38, 48)  # very dark teal
for i in range(overlay_height):
    alpha = int(50 * (1 - (i / max(overlay_height - 1, 1))))  # fading effect
    # blend by computing intermediate color between banner bottom and overlay_color
    t = (i / max(overlay_height - 1, 1)) * 0.6
    r = int((1 - t) * end_color[0] + t * overlay_color[0])
    g = int((1 - t) * end_color[1] + t * overlay_color[1])
    b = int((1 - t) * end_color[2] + t * overlay_color[2])
    draw.line([(0, banner_bottom - overlay_height + i), (W, banner_bottom - overlay_height + i)], fill=(r, g, b))

# 4) Main content background (white) below banner
content_top = banner_bottom
draw.rectangle([(0, content_top), (W, H)], fill="#FFFFFF")

# 5) Main organizer/info card (rounded rectangle) behind profile + follow button
card_margin_x = 40
card_top = 1200
card_bottom = 1360
card_radius = 28
card_box = [card_margin_x, card_top, W - card_margin_x, card_bottom]

# Subtle drop shadow for the card (drawn as a faint larger rounded rect behind)
shadow_offset = 8
shadow_box = [card_box[0] + shadow_offset, card_box[1] + shadow_offset,
              card_box[2] + shadow_offset, card_box[3] + shadow_offset]
draw.rounded_rectangle(shadow_box, radius=card_radius + 4, fill="#E9E9EE")

# Card fill
draw.rounded_rectangle(card_box, radius=card_radius, fill="#F6F6F9", outline="#E6E6EB", width=1)

# 6) Thin separators between sections
sep_color = "#E6E6EB"
# Separator under the profile card area
draw.line([(40, card_bottom + 80), (W - 40, card_bottom + 80)], fill=sep_color, width=1)

# Separator under event details area (approx)
draw.line([(40, 1680), (W - 40, 1680)], fill=sep_color, width=1)

# Separator above "About this event" area
draw.line([(40, 2000), (W - 40, 2000)], fill=sep_color, width=1)

# 7) Light section header background strip (for "About this event" area)
about_top = 2040
about_height = 140
draw.rectangle([(0, about_top - 20), (W, about_top + about_height)], fill="#FFFFFF")

# 8) Tag/metadata card background (a faint rounded area behind small badges)
# Place a large pale rounded container (but avoid drawing over detected small icons)
meta_box = [40, 2100, W - 40, 2200]
draw.rounded_rectangle(meta_box, radius=18, fill="#FFFFFF", outline="#F0F0F3", width=1)

# 9) Location section divider area background (subtle)
location_top = 2640
draw.rectangle([(0, location_top - 10), (W, H)], fill="#FFFFFF")
draw.line([(40, location_top + 40), (W - 40, location_top + 40)], fill=sep_color, width=1)

# 10) Floating bottom content area hint (subtle pale rounded card behind final controls)
bottom_hint_top = H - 220
draw.rounded_rectangle([40, bottom_hint_top, W - 40, H - 40], radius=22, fill="#FFFFFF", outline="#EAEAF0", width=1)

# 11) Decorative left edge margin vertical rule for content separation
draw.line([(40, content_top + 40), (40, H - 200)], fill="#F5F5F7", width=6)

# Note: All textual labels, icons, and interactive controls are intentionally NOT drawn here.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/01_icon_career_fairs.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["career_fairs"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/02_icon_Business_Professional.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2427), _c2)
except Exception:
    pass
layout["Business_&_Professional"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/04_icon_Ticket_sales_end_soon.png
try:
    _c4 = get_crop(4, 547, 84)
    canvas.paste(_c4, (40, 753), _c4)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/05_icon_9.32.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["9.32"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 63)
    canvas.paste(_c6, (1318, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1318, 1, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 94, 61)
    canvas.paste(_c7, (1215, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 2, 1309, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/08_icon_Show_map.png
try:
    _c8 = get_crop(8, 226, 144)
    canvas.paste(_c8, (1166, 2645), _c8)
except Exception:
    pass
layout["Show_map"] = [1166, 2645, 1392, 2789]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/09_icon_NEW_YORK.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1116, 108), _c9)
except Exception:
    pass
layout["NEW_YORK"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 58)
    canvas.paste(_c10, (316, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 5, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/11_icon_Are_you_looking_for_a_job_in_New_York_If.png
try:
    _c11 = get_crop(11, 234, 144)
    canvas.paste(_c11, (48, 2427), _c11)
except Exception:
    pass
layout["Are_you_looking_for_a_job"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/12_icon_9.32.png
try:
    _c12 = get_crop(12, 54, 63)
    canvas.paste(_c12, (116, 2), _c12)
except Exception:
    pass
layout["9.32"] = [116, 2, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/13_icon_New_York.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (96, 1289), _c13)
except Exception:
    pass
layout["New_York"] = [96, 1289, 240, 1433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/14_text_9.32.png
try:
    _c14 = get_crop(14, 91, 43)
    canvas.paste(_c14, (20, 17), _c14)
except Exception:
    pass
layout["9.32"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/15_text_New_York_Job_Fair_March_21_2024.png
try:
    _c15 = get_crop(15, 468, 144)
    canvas.paste(_c15, (288, 1250), _c15)
except Exception:
    pass
layout["New_York_Job_Fair_March_2"] = [288, 1250, 756, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/16_text_New.png
try:
    _c16 = get_crop(16, 167, 74)
    canvas.paste(_c16, (1204, 1017), _c16)
except Exception:
    pass
layout["New"] = [1204, 1017, 1371, 1091]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/17_text_York_Career_Fairs.png
try:
    _c17 = get_crop(17, 468, 144)
    canvas.paste(_c17, (288, 1250), _c17)
except Exception:
    pass
layout["York_Career_Fairs"] = [288, 1250, 756, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/18_text_Best_Hire_Career_Fairs.png
try:
    _c18 = get_crop(18, 468, 144)
    canvas.paste(_c18, (288, 1250), _c18)
except Exception:
    pass
layout["Best_Hire_Career_Fairs"] = [288, 1250, 756, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/19_text_30.9k_Followers.png
try:
    _c19 = get_crop(19, 468, 144)
    canvas.paste(_c19, (288, 1250), _c19)
except Exception:
    pass
layout["30.9k_Followers"] = [288, 1250, 756, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/20_text_New_York.png
try:
    _c20 = get_crop(20, 202, 49)
    canvas.paste(_c20, (141, 1566), _c20)
except Exception:
    pass
layout["New_York"] = [141, 1566, 343, 1615]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/21_text_4hrs.png
try:
    _c21 = get_crop(21, 112, 50)
    canvas.paste(_c21, (141, 1674), _c21)
except Exception:
    pass
layout["4hrs"] = [141, 1674, 253, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/22_text_Refund_policy.png
try:
    _c22 = get_crop(22, 299, 63)
    canvas.paste(_c22, (138, 1780), _c22)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/23_text_The_organizer_will_review_refund_request.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1517), _c23)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/24_text_About_this_event.png
try:
    _c24 = get_crop(24, 452, 61)
    canvas.paste(_c24, (45, 2080), _c24)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 497, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/25_text_Location.png
try:
    _c25 = get_crop(25, 246, 61)
    canvas.paste(_c25, (41, 2691), _c25)
except Exception:
    pass
layout["Location"] = [41, 2691, 287, 2752]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/26_text_New_York.png
try:
    _c26 = get_crop(26, 209, 49)
    canvas.paste(_c26, (141, 2817), _c26)
except Exception:
    pass
layout["New_York"] = [141, 2817, 350, 2866]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_11_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-13/27_text_New_York_Virtual_Job_Fair_New_York_NY_10.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 2427), _c27)
except Exception:
    pass
layout["New_York;_Virtual_Job_Fai"] = [48, 2427, 282, 2571]
