# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_08
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10.png
# step_index: 8/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for 1440x2960 canvas
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (ensure clean white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Top status bar
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#D9D9D9")  # light gray status bar

# Header / toolbar area (below status bar)
toolbar_top = status_h
toolbar_bottom = 232
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")

# Toolbar bottom divider
draw.line([(24, toolbar_bottom), (1416, toolbar_bottom)], fill="#E8E6EA", width=1)

# Main content subtle card background behind sections (full width with slight inset)
content_inset = 24
# First large content block (contains event meta and about)
block1_top = toolbar_bottom + 24
block1_bottom = 1280
draw.rectangle(
    [(content_inset, block1_top), (1440 - content_inset, block1_bottom)],
    fill="#FFFFFF"
)

# Light separators inside the first content block to separate policy / about / location areas
sep_color = "#ECE8EE"
# Separator under refund/policy area (approx)
draw.line([(36, 420), (1404, 420)], fill=sep_color, width=1)
# Separator under "About this event" area
draw.line([(36, 820), (1404, 820)], fill=sep_color, width=1)
# Separator above Location section
draw.line([(36, 1150), (1404, 1150)], fill=sep_color, width=1)

# Location area background subtle (keeps consistent spacing)
loc_box_top = 1158
loc_box_bottom = 1330
draw.rectangle([(36, loc_box_top), (1404, loc_box_bottom)], fill="#FFFFFF")

# FAQ section area (separate block)
faq_top = 1360
faq_bottom = 2320
draw.rectangle([(content_inset, faq_top), (1440 - content_inset, faq_bottom)], fill="#FFFFFF")

# FAQ separators (thin lines between items)
faq_sep_y = [1700, 1970, 2190]
for y in faq_sep_y:
    draw.line([(36, y), (1404, y)], fill=sep_color, width=1)

# Large subtle section title spacing indicator (no text, just vertical spacing background hints)
# A faint left accent bar for headings (helps visually separate sections)
accent_x = 36
accent_w = 6
draw.rectangle([(accent_x, 300), (accent_x + accent_w, 370)], fill="#F6EEF8")
draw.rectangle([(accent_x, 880), (accent_x + accent_w, 940)], fill="#F6EEF8")
draw.rectangle([(accent_x, 1360), (accent_x + accent_w, 1400)], fill="#F6EEF8")

# Bottom fixed ticket bar background
bottom_bar_top = 2680
bottom_bar_bottom = 2960
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_bottom)], fill="#F6F4F6")
# Top border for bottom bar
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill="#E6E0E6", width=1)

# Slight rounded panel behind floating deal area (do not draw deal text or icon)
# Keep shape generic and offset from detected deal area so we do not duplicate the item
# (draw only a very faint rounded rectangle as background hint)
deal_bg_box = (1000, 2660, 1380, 2820)
draw.rounded_rectangle(deal_bg_box, radius=18, fill="#FFF6EA", outline="#F1D9B8", width=1)

# Subtle drop-shadow lines under major sections for depth (very faint)
shadow_color = (0, 0, 0, 6)  # not used directly (ImageDraw doesn't support alpha here), emulate with very light gray
draw.line([(36, block1_bottom + 2), (1404, block1_bottom + 2)], fill="#F3F2F4", width=1)
draw.line([(36, faq_bottom + 2), (1404, faq_bottom + 2)], fill="#F3F2F4", width=1)

# Right-side thin divider near edges (visual balance)
draw.line([(1404, toolbar_top + 8), (1404, toolbar_bottom - 8)], fill="#F0EEF1", width=1)

# Decorative faint horizontal grid to suggest content rows (very light, non-intrusive)
for y in range(520, 2200, 220):
    draw.line([(48, y), (1392, y)], fill="#FBFAFB", width=1)

# Final refinement - small bottom inset line to separate page from device bezel
draw.line([(0, 2958), (1440, 2958)], fill="#EAE6EA", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/02_icon_Get_tickets.png
try:
    _c2 = get_crop(2, 570, 144)
    canvas.paste(_c2, (822, 2768), _c2)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/03_icon_9.26.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.26"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 57)
    canvas.paste(_c4, (316, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 5, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/05_icon_Performing_Visual_Arts_._Comedy.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 1072), _c5)
except Exception:
    pass
layout["Performing_&_Visual_Arts_"] = [48, 1072, 282, 1216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 58)
    canvas.paste(_c6, (247, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [247, 4, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 97, 59)
    canvas.paste(_c7, (1216, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1216, 2, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/08_icon_Von_3_Bleecker_Street_New_York_NY_10012.png
try:
    _c8 = get_crop(8, 226, 144)
    canvas.paste(_c8, (1166, 1290), _c8)
except Exception:
    pass
layout["Von,_3_Bleecker_Street;_N"] = [1166, 1290, 1392, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/09_icon_9.26.png
try:
    _c9 = get_crop(9, 56, 58)
    canvas.paste(_c9, (180, 4), _c9)
except Exception:
    pass
layout["9.26"] = [180, 4, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/10_icon_Von_is_a_bar_Speakeasy_Where_is_the_show.png
try:
    _c10 = get_crop(10, 1440, 588)
    canvas.paste(_c10, (0, 1882), _c10)
except Exception:
    pass
layout["Von_is_a_bar?_Speakeasy??"] = [0, 1882, 1440, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/11_icon_1_hrs_30_mins.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (36, 108), _c11)
except Exception:
    pass
layout["1_hrs_30_mins"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 59)
    canvas.paste(_c12, (1318, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1318, 2, 1372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/13_icon_9.26.png
try:
    _c13 = get_crop(13, 52, 59)
    canvas.paste(_c13, (117, 3), _c13)
except Exception:
    pass
layout["9.26"] = [117, 3, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/14_icon_Refund_policy.png
try:
    _c14 = get_crop(14, 373, 75)
    canvas.paste(_c14, (58, 416), _c14)
except Exception:
    pass
layout["Refund_policy"] = [58, 416, 431, 491]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/15_icon_Show_map.png
try:
    _c15 = get_crop(15, 226, 144)
    canvas.paste(_c15, (1166, 1290), _c15)
except Exception:
    pass
layout["Show_map"] = [1166, 1290, 1392, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/16_icon_2_for_1_deal.png
try:
    _c16 = get_crop(16, 570, 144)
    canvas.paste(_c16, (822, 2768), _c16)
except Exception:
    pass
layout["2_for_1_deal"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/17_icon_The_show_is_sold_out_but_my_friend_wants.png
try:
    _c17 = get_crop(17, 1440, 588)
    canvas.paste(_c17, (0, 1882), _c17)
except Exception:
    pass
layout["The_show_is_sold_out_but_"] = [0, 1882, 1440, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/18_icon_The_best_comedy_show_in_the_East_Village.png
try:
    _c18 = get_crop(18, 234, 144)
    canvas.paste(_c18, (48, 1072), _c18)
except Exception:
    pass
layout["The_best_comedy_show_in_t"] = [48, 1072, 282, 1216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/19_text_9.26.png
try:
    _c19 = get_crop(19, 91, 45)
    canvas.paste(_c19, (20, 15), _c19)
except Exception:
    pass
layout["9.26"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/20_text_The_Good_Mood_Comedy-.png
try:
    _c20 = get_crop(20, 1344, 42)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["The_Good_Mood_Comedy-"] = [48, 264, 1392, 306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 42)
    canvas.paste(_c21, (48, 264), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 264, 1392, 306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/22_text_Location.png
try:
    _c22 = get_crop(22, 246, 64)
    canvas.paste(_c22, (41, 1333), _c22)
except Exception:
    pass
layout["Location"] = [41, 1333, 287, 1397]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/23_text_FAQs.png
try:
    _c23 = get_crop(23, 158, 73)
    canvas.paste(_c23, (41, 1741), _c23)
except Exception:
    pass
layout["FAQs"] = [41, 1741, 199, 1814]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_08_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-10/24_text_S0_-_8.24.png
try:
    _c24 = get_crop(24, 242, 61)
    canvas.paste(_c24, (89, 2811), _c24)
except Exception:
    pass
layout["S0_-_$8.24"] = [89, 2811, 331, 2872]
