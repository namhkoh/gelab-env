# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_10
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12.png
# step_index: 10/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (RGB 1440x2960), fonts: font_sm, font_md, font_lg, font_xl
w, h = canvas.size

# Colors (picked to match the screenshot's soft, cool UI)
bg_main = (250, 250, 252)       # very light off-white background
status_bar = (198, 198, 198)    # light gray status bar
hero_bg = (236, 243, 249)       # pale bluish hero background
content_white = (255, 255, 255) # main content white
card_fill = (250, 249, 253)     # subtle off-white card
card_outline = (238, 233, 243)  # very light purple/gray outline for cards
divider = (236, 236, 239)       # thin separator color
soft_shadow = (230, 231, 235)   # soft shadow line

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_main)

# Status bar area (top ~50px)
status_h = 60
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar)

# Hero image background area (under the status bar)
hero_top = status_h
hero_bottom = 460
draw.rectangle([(0, hero_top), (w, hero_bottom)], fill=hero_bg)

# Add a subtle darker band near the top edge of the hero to simulate app header fade
draw.rectangle([(0, hero_top), (w, hero_top + 14)], fill=(224, 230, 235))

# Add a soft bottom edge shadow under the hero area
draw.rectangle([(0, hero_bottom - 6), (w, hero_bottom)], fill=soft_shadow)

# Main white content area below hero
content_top = hero_bottom
draw.rectangle([(0, content_top), (w, h)], fill=content_white)

# Organizer / host card (rounded) behind the "Wayne Squires" and "Follow" elements.
# Keep padding consistent with screenshot margins
card_x0 = 48
card_x1 = w - 48
card_y0 = 1195
card_y1 = card_y0 + 144
card_radius = 28
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# Add subtle shadow under the organizer card
shadow_y = card_y1 + 6
draw.rectangle([(card_x0, card_y1), (card_x1, shadow_y)], fill=soft_shadow)

# Horizontal separators between major sections
# Separator below refund policy / event details area (~under the small descriptive text)
sep_y1 = 1740
draw.line([(48, sep_y1), (w - 48, sep_y1)], fill=divider, width=1)

# Separator under "About this event" content block
sep_y2 = 2120
draw.line([(48, sep_y2), (w - 48, sep_y2)], fill=divider, width=1)

# Separator above Location block (near bottom region)
sep_y3 = 2540
draw.line([(48, sep_y3), (w - 48, sep_y3)], fill=divider, width=1)

# Light rounded container for the "About this event" area background (very subtle)
about_x0 = 40
about_x1 = w - 40
about_y0 = 1920
about_y1 = 2240
draw.rounded_rectangle([(about_x0, about_y0), (about_x1, about_y1)],
                       radius=20, fill=content_white, outline=None)

# Subtle left accent line for the Location section to anchor it visually
loc_block_y0 = 2560
loc_block_y1 = 2920
accent_x = 48
draw.line([(accent_x, loc_block_y0), (accent_x, loc_block_y1)], fill=(243, 241, 247), width=8)

# Small faint divider near the top of the content to separate title area from hero (under title)
title_sep_y = 860
draw.line([(48, title_sep_y), (w - 48, title_sep_y)], fill=(247, 247, 249), width=1)

# Large subtle divider band near mid-page to mimic content grouping
band_y0 = 1500
band_y1 = band_y0 + 6
draw.rectangle([(0, band_y0), (w, band_y1)], fill=(250, 250, 251))

# Add a faint rounded background behind the "Show map" area (right side) to indicate interactive area
map_bg_w = 320
map_bg_h = 120
map_bg_x1 = w - 48
map_bg_x0 = map_bg_x1 - map_bg_w
map_bg_y1 = 2520
map_bg_y0 = map_bg_y1 - map_bg_h
draw.rounded_rectangle([(map_bg_x0, map_bg_y0), (map_bg_x1, map_bg_y1)],
                       radius=14, fill=(255, 255, 255), outline=(240, 240, 244))

# Final subtle top-left back-button area background (behind the circular back icon)
back_button_center = (64, status_h + 64)
back_button_r = 44
draw.ellipse([(back_button_center[0] - back_button_r, back_button_center[1] - back_button_r),
              (back_button_center[0] + back_button_r, back_button_center[1] + back_button_r)],
             fill=(255, 255, 255), outline=(240, 240, 244))

# Done drawing background structure and separators. UI content (icons/text) will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/01_icon_Just_addedl.png
try:
    _c1 = get_crop(1, 313, 144)
    canvas.paste(_c1, (48, 724), _c1)
except Exception:
    pass
layout["Just_addedl"] = [48, 724, 361, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/02_icon_Sports_Fitness.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2332), _c2)
except Exception:
    pass
layout["Sports_&_Fitness"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 47, 66)
    canvas.paste(_c3, (1156, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [1156, 2, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/04_icon_8.11.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["8.11"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/05_icon_More.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/06_icon_Share.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 108), _c6)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/07_icon_8.11.png
try:
    _c7 = get_crop(7, 66, 71)
    canvas.paste(_c7, (111, 0), _c7)
except Exception:
    pass
layout["8.11"] = [111, 0, 177, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 62)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 73, 71)
    canvas.paste(_c9, (305, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [305, 1, 378, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/10_icon_8.11.png
try:
    _c10 = get_crop(10, 68, 71)
    canvas.paste(_c10, (177, 1), _c10)
except Exception:
    pass
layout["8.11"] = [177, 1, 245, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 56, 71)
    canvas.paste(_c11, (246, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 1, 302, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/12_icon_Wayne_Squires.png
try:
    _c12 = get_crop(12, 320, 144)
    canvas.paste(_c12, (144, 1195), _c12)
except Exception:
    pass
layout["Wayne_Squires"] = [144, 1195, 464, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/13_icon_Show_map.png
try:
    _c13 = get_crop(13, 226, 144)
    canvas.paste(_c13, (1166, 2550), _c13)
except Exception:
    pass
layout["Show_map"] = [1166, 2550, 1392, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/14_icon_hrs_30_mins.png
try:
    _c14 = get_crop(14, 315, 75)
    canvas.paste(_c14, (115, 1567), _c14)
except Exception:
    pass
layout["hrs_30_mins"] = [115, 1567, 430, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 100, 63)
    canvas.paste(_c15, (1215, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1215, 2, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/16_icon_8.11.png
try:
    _c16 = get_crop(16, 110, 71)
    canvas.paste(_c16, (2, 0), _c16)
except Exception:
    pass
layout["8.11"] = [2, 0, 112, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/17_text_Thursday_May_2_6_00_PM.png
try:
    _c17 = get_crop(17, 313, 144)
    canvas.paste(_c17, (48, 724), _c17)
except Exception:
    pass
layout["Thursday;_May_2_*_6:00_PM"] = [48, 724, 361, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/18_text_Outdoor_HIIT.png
try:
    _c18 = get_crop(18, 320, 144)
    canvas.paste(_c18, (144, 1195), _c18)
except Exception:
    pass
layout["Outdoor_HIIT"] = [144, 1195, 464, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/19_text_Jackson_Playground.png
try:
    _c19 = get_crop(19, 1344, 144)
    canvas.paste(_c19, (48, 1422), _c19)
except Exception:
    pass
layout["Jackson_Playground"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/20_text_Refund_policy.png
try:
    _c20 = get_crop(20, 299, 63)
    canvas.paste(_c20, (138, 1685), _c20)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1422), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/22_text_About_this_event.png
try:
    _c22 = get_crop(22, 453, 67)
    canvas.paste(_c22, (44, 1982), _c22)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/23_text_Get_ready_to_sweat_and_push_your_limits_.png
try:
    _c23 = get_crop(23, 234, 144)
    canvas.paste(_c23, (48, 2332), _c23)
except Exception:
    pass
layout["Get_ready_to_sweat_and_pu"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/24_text_Read_more.png
try:
    _c24 = get_crop(24, 234, 144)
    canvas.paste(_c24, (48, 2332), _c24)
except Exception:
    pass
layout["Read_more"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/25_text_Location.png
try:
    _c25 = get_crop(25, 244, 63)
    canvas.paste(_c25, (43, 2594), _c25)
except Exception:
    pass
layout["Location"] = [43, 2594, 287, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/26_text_Jackson_Playground.png
try:
    _c26 = get_crop(26, 437, 63)
    canvas.paste(_c26, (136, 2721), _c26)
except Exception:
    pass
layout["Jackson_Playground"] = [136, 2721, 573, 2784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/27_text_Jackson_Playground_Jackson_Playground_Sa.png
try:
    _c27 = get_crop(27, 226, 144)
    canvas.paste(_c27, (1166, 2550), _c27)
except Exception:
    pass
layout["Jackson_Playground,_Jacks"] = [1166, 2550, 1392, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_10_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-12/28_text_CA_94107.png
try:
    _c28 = get_crop(28, 225, 52)
    canvas.paste(_c28, (139, 2851), _c28)
except Exception:
    pass
layout["CA_94107"] = [139, 2851, 364, 2903]
