# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_08
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10.png
# step_index: 8/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background (canvas already provided)
draw.rectangle([(0, 0), canvas.size], fill="#FFFFFF")

# Status bar area (top ~72px) - subtle gray bar
status_bar_h = 72
draw.rectangle([(0, 0), (canvas.width, status_bar_h)], fill="#CFCFCF")

# Header / search area background (below status bar)
header_top = status_bar_h
header_bottom = 220
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill="#FFFFFF")

# Blue underline for the search field (thin accent line)
underline_y = header_bottom - 12
underline_margin_x = 48
draw.rectangle([(underline_margin_x, underline_y), (canvas.width - underline_margin_x, underline_y + 6)], fill="#2E5BFF")

# subtle divider below header
draw.line([(0, header_bottom), (canvas.width, header_bottom)], fill="#E6E7EA", width=1)

# "Popular" list separators (these are just subtle dividing lines between the list rows)
popular_divider_positions = [378 + 120, 498 + 120, 618 + 120, 738 + 120, 858 + 120]
for y in popular_divider_positions:
    # limit to canvas
    if 0 < y < canvas.height - 200:
        draw.line([(48, y), (canvas.width - 48, y)], fill="#F0F0F2", width=1)

# Section header divider before Events section
events_section_y = 1117
draw.line([(48, events_section_y - 32), (canvas.width - 48, events_section_y - 32)], fill="#EFEFF1", width=1)

# Event list card backgrounds (rounded rectangles behind each event row)
# Using detected event y positions and sizes: 1117, 1513, 1909, 2305 (height ~396)
event_rows = [1117, 1513, 1909, 2305]
card_margin_x = 48
card_width = canvas.width - card_margin_x * 2
card_h = 396
card_fill = "#FFFFFF"
card_outline = "#EFEFF2"
for y in event_rows:
    top = y
    bottom = y + card_h
    # Keep card inside canvas bounds
    if top < canvas.height and bottom > 0:
        draw.rounded_rectangle(
            [(card_margin_x, top), (card_margin_x + card_width, min(bottom, canvas.height - 200))],
            radius=10,
            fill=card_fill,
            outline=card_outline,
            width=1
        )

# Thin separators between event cards
for y in event_rows:
    sep_y = y + card_h + 8
    if sep_y < canvas.height - 200:
        draw.line([(card_margin_x, sep_y), (canvas.width - card_margin_x, sep_y)], fill="#F4F4F6", width=1)

# Left thumbnail placeholder backgrounds for event list (subtle neutral rectangles behind where thumbnails will be pasted)
# Keep these minimal (no icons/text) so pasted thumbnails overlay naturally.
thumb_w = 144
thumb_h = 132
thumb_x = card_margin_x
for y in event_rows:
    top = y + 22
    bottom = top + thumb_h
    if bottom < canvas.height:
        draw.rectangle([(thumb_x, top), (thumb_x + thumb_w, bottom)], fill="#F6F7F8", outline="#EDEFF1", width=1)

# Bottom navigation bar area (approx 156px high at bottom)
nav_top = 2804
nav_bottom = canvas.height
draw.rectangle([(0, nav_top), (canvas.width, nav_bottom)], fill="#FFFFFF")
# top border for nav bar
draw.line([(0, nav_top), (canvas.width, nav_top)], fill="#E8E8EA", width=1)

# subtle highlight behind center-search nav item area (to match screenshot accent)
center_accent_y0 = nav_top + 8
center_accent_y1 = nav_top + 8 + 4
draw.rectangle([(0, center_accent_y0), (canvas.width, center_accent_y1)], fill="#FF6A00", outline=None)

# Final subtle overall vignette/shadow under header for depth
shadow_top = header_bottom
for i, alpha in enumerate([12, 8, 6]):
    y = shadow_top + i
    # very subtle darker line
    draw.line([(0, y), (canvas.width, y)], fill=(220, 221, 224, alpha) if canvas.mode == "RGBA" else "#EAEAF0")

# End of background and structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/00_icon_Outdoor.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Outdoor]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/01_icon_Online.png
try:
    _c1 = get_crop(1, 111, 48)
    canvas.paste(_c1, (391, 1386), _c1)
except Exception:
    pass
layout["Online"] = [391, 1386, 502, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/02_icon_Nor_Cal_Outdoor_Academy.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1909), _c2)
except Exception:
    pass
layout["Nor_Cal_Outdoor_Academy"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/03_icon_8.11.png
try:
    _c3 = get_crop(3, 55, 60)
    canvas.paste(_c3, (115, 3), _c3)
except Exception:
    pass
layout["8.11"] = [115, 3, 170, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/04_icon_Outdoor.png
try:
    _c4 = get_crop(4, 56, 59)
    canvas.paste(_c4, (313, 4), _c4)
except Exception:
    pass
layout["Outdoor]"] = [313, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/05_icon_8.11.png
try:
    _c5 = get_crop(5, 52, 59)
    canvas.paste(_c5, (184, 3), _c5)
except Exception:
    pass
layout["8.11"] = [184, 3, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 42, 54)
    canvas.paste(_c6, (254, 7), _c6)
except Exception:
    pass
layout["icon_6"] = [254, 7, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/07_icon_8.11.png
try:
    _c7 = get_crop(7, 117, 105)
    canvas.paste(_c7, (57, 118), _c7)
except Exception:
    pass
layout["8.11"] = [57, 118, 174, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 40, 66)
    canvas.paste(_c8, (1159, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1159, 1, 1199, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/09_icon_Cancel.png
try:
    _c9 = get_crop(9, 92, 65)
    canvas.paste(_c9, (1218, 0), _c9)
except Exception:
    pass
layout["Cancel"] = [1218, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/10_icon_Sat_Apr_27.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (288, 2804), _c10)
except Exception:
    pass
layout["Sat,_Apr_27__"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/11_icon_6_00_PM_PDT.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1513), _c11)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/12_icon_Great_Outdoors.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1117), _c12)
except Exception:
    pass
layout["Great_Outdoors"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/13_icon_Tickets.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (864, 2804), _c13)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 46, 61)
    canvas.paste(_c14, (1323, 2), _c14)
except Exception:
    pass
layout["Cancel"] = [1323, 2, 1369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/15_icon_1I_O0AM_PDT.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["1I:O0AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/16_icon_227_creator_followers.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 2305), _c16)
except Exception:
    pass
layout["227_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/17_icon_6_00_PM_PDT.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1513), _c17)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1099, 96), _c18)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/19_icon_outdoor_festival.png
try:
    _c19 = get_crop(19, 1344, 120)
    canvas.paste(_c19, (48, 738), _c19)
except Exception:
    pass
layout["outdoor_festival"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/20_icon_Latina_Outaoors_SF_Bay_Arca.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 2305), _c20)
except Exception:
    pass
layout["Latina_Outaoors_|_SF_Bay_"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/21_icon_Events.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1117), _c21)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/22_icon_Outdoor_HIIT.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 1513), _c22)
except Exception:
    pass
layout["Outdoor_HIIT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/23_icon_12_O0AM_EDT.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1117), _c23)
except Exception:
    pass
layout["12:O0AM_EDT"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/24_icon_outdoor_concert.png
try:
    _c24 = get_crop(24, 1344, 120)
    canvas.paste(_c24, (48, 378), _c24)
except Exception:
    pass
layout["outdoor_concert"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 149, 144)
    canvas.paste(_c25, (1243, 97), _c25)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/26_icon_8_00_AM_PDT.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1909), _c26)
except Exception:
    pass
layout["8:00_AM_PDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/27_icon_outdoor_events.png
try:
    _c27 = get_crop(27, 1344, 120)
    canvas.paste(_c27, (48, 498), _c27)
except Exception:
    pass
layout["outdoor_events"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/28_icon_Outdoor_Adventure_Retreat_Embrace_the.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1117), _c28)
except Exception:
    pass
layout["Outdoor_Adventure_Retreat"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/29_icon_8.11.png
try:
    _c29 = get_crop(29, 94, 61)
    canvas.paste(_c29, (13, 2), _c29)
except Exception:
    pass
layout["8.11"] = [13, 2, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/30_icon_SFMII_ITA_S_niitnonds.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["SFMII_ITA_S_niitnonds"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/31_icon_Dance_Outdoors_with_Rhythm_Motion.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2305), _c31)
except Exception:
    pass
layout["Dance_Outdoors_with_Rhyth"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/32_icon_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/33_text_Popular.png
try:
    _c33 = get_crop(33, 224, 78)
    canvas.paste(_c33, (41, 298), _c33)
except Exception:
    pass
layout["Popular"] = [41, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/34_text_bernal_heights_outdoor_cinema.png
try:
    _c34 = get_crop(34, 1344, 120)
    canvas.paste(_c34, (48, 618), _c34)
except Exception:
    pass
layout["bernal_heights_outdoor_ci"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/35_text_Latina_Outaoors_SF_Bay_Arca.png
try:
    _c35 = get_crop(35, 162, 16)
    canvas.paste(_c35, (116, 2762), _c35)
except Exception:
    pass
layout["Latina_Outaoors_|_SF_Bay_"] = [116, 2762, 278, 2778]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/36_text_Sat_Apr_27.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (288, 2804), _c36)
except Exception:
    pass
layout["Sat,_Apr_27__"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/37_text_1I_O0AM_PDT.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (576, 2804), _c37)
except Exception:
    pass
layout["1I:O0AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/38_text_SFMII_ITA_S_niitnonds.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (0, 2804), _c38)
except Exception:
    pass
layout["SFMII_ITA_S_niitnonds"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_08_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-10/39_clickable_outdoor_yoga.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 858), _c39)
except Exception:
    pass
layout["outdoor_yoga"] = [48, 858, 1392, 1002]
