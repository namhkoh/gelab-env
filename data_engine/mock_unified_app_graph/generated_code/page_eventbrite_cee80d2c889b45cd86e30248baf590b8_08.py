# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_08
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10.png
# step_index: 8/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base
draw.rectangle((0, 0, 1440, 2960), fill="#FBFCFF")  # page background (very light)

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#9A9A9A")  # status bar background (muted gray)

# Header/search area background (under status bar)
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")  # white header/search field background

# Blue underline for search field (prominent accent)
underline_y = header_bottom - 16
draw.rectangle((48, underline_y, 1440 - 48, underline_y + 6), fill="#264EDB")  # blue underline

# Subtle shadow under header
draw.rectangle((0, header_bottom, 1440, header_bottom + 3), fill="#E9EAF0")

# "Events" section background strip (behind the section title)
events_strip_top = 280
events_strip_bottom = 340
draw.rounded_rectangle((40, events_strip_top, 1400, events_strip_bottom), radius=8, fill="#FFFFFF", outline=None)

# Card-like rounded backgrounds for each listed event (leave content area empty for auto-pasted elements)
card_x = 48
card_w = 1344
card_h = 396
card_radius = 12
card_ys = [390, 786, 1182, 1578, 1974, 2370]

for y in card_ys:
    # subtle drop shadow (offset)
    shadow_offset = 6
    draw.rounded_rectangle(
        (card_x, y + shadow_offset, card_x + card_w, y + card_h + shadow_offset),
        radius=card_radius + 2,
        fill="#F2F4F8",
        outline=None
    )
    # main card surface
    draw.rounded_rectangle(
        (card_x, y, card_x + card_w, y + card_h),
        radius=card_radius,
        fill="#FFFFFF",
        outline="#E6E7EB",
        width=1
    )
    # thin divider line at bottom of card
    draw.line((card_x + 8, y + card_h - 1, card_x + card_w - 8, y + card_h - 1), fill="#F0F1F4", width=1)

# Separator line between list area and bottom navigation
nav_top = 2800
draw.line((24, nav_top, 1440 - 24, nav_top), fill="#E6E7EB", width=2)

# Bottom navigation background bar
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
# subtle top shadow for nav
draw.rectangle((0, nav_top, 1440, nav_top + 3), fill="#EDEFF3")

# Edge gutters (subtle)
draw.rectangle((0, 0, 24, 2960), fill="#FBFCFF")
draw.rectangle((1440 - 24, 0, 1440, 2960), fill="#FBFCFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/00_icon_sal11.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1974), _c0)
except Exception:
    pass
layout["sal11"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/01_icon_dolce.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2370), _c1)
except Exception:
    pass
layout["dolce"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/02_icon_Food_Drink.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/03_icon_II_Bacco.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 786), _c3)
except Exception:
    pass
layout["II_Bacco"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/04_icon_MieM.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1578), _c4)
except Exception:
    pass
layout["MieM"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/05_icon_E7.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 390), _c5)
except Exception:
    pass
layout["E7"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/06_icon_9.44.png
try:
    _c6 = get_crop(6, 52, 60)
    canvas.paste(_c6, (183, 2), _c6)
except Exception:
    pass
layout["9.44"] = [183, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 58)
    canvas.paste(_c7, (249, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 4, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/08_icon_Cantina_Vibes_Food_Drinks_Hookah.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1974), _c8)
except Exception:
    pass
layout["Cantina_Vibes,_Food,_Drin"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/09_icon_9.44.png
try:
    _c9 = get_crop(9, 55, 60)
    canvas.paste(_c9, (114, 3), _c9)
except Exception:
    pass
layout["9.44"] = [114, 3, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 42, 66)
    canvas.paste(_c10, (1158, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1158, 1, 1200, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 58)
    canvas.paste(_c11, (316, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [316, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/12_icon_kaida.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1182), _c12)
except Exception:
    pass
layout["kaida"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/13_icon_II_Bacco.png
try:
    _c13 = get_crop(13, 132, 49)
    canvas.paste(_c13, (391, 1054), _c13)
except Exception:
    pass
layout["II_Bacco"] = [391, 1054, 523, 1103]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 73, 64)
    canvas.paste(_c14, (1217, 1), _c14)
except Exception:
    pass
layout["Cancel"] = [1217, 1, 1290, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/15_icon_9.44.png
try:
    _c15 = get_crop(15, 119, 104)
    canvas.paste(_c15, (56, 118), _c15)
except Exception:
    pass
layout["9.44"] = [56, 118, 175, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 50, 63)
    canvas.paste(_c16, (1320, 1), _c16)
except Exception:
    pass
layout["Cancel"] = [1320, 1, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/17_icon_The_Original_E._Village_Food_Drinks.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1182), _c17)
except Exception:
    pass
layout["The_Original_E._Village_F"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 149, 144)
    canvas.paste(_c18, (1243, 97), _c18)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/19_icon_Tickets.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1099, 96), _c20)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/21_icon_8_159_creator_followers.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["8_159_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/22_icon_8_159_creator_followers.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["8_159_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/23_icon_Food_Drink.png
try:
    _c23 = get_crop(23, 46, 59)
    canvas.paste(_c23, (384, 3), _c23)
except Exception:
    pass
layout["Food_&_Drink"] = [384, 3, 430, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/24_icon_Uptown_Vibes_Food_Drinks_Hookah.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1578), _c24)
except Exception:
    pass
layout["Uptown_Vibes,_Food,_Drink"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/25_icon_Grand_Opening_Party_for_Oak_Knowledge.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 786), _c25)
except Exception:
    pass
layout["Grand_Opening_Party_for_O"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/26_icon_LIVE_DJ_2_ROOMS_PATIO.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 2370), _c26)
except Exception:
    pass
layout["LIVE_DJ_2_ROOMS_+_PATIO"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/27_icon_Skinnys_CAntina_on_the_Hudson.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1578), _c27)
except Exception:
    pass
layout["Skinnys_CAntina_on_the_Hu"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/28_icon_dolce.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["dolce"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/29_icon_Realty_School_Dance.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 786), _c29)
except Exception:
    pass
layout["Realty_School_(Dance"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/30_icon_Cancel.png
try:
    _c30 = get_crop(30, 43, 64)
    canvas.paste(_c30, (1271, 1), _c30)
except Exception:
    pass
layout["Cancel"] = [1271, 1, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/31_text_9.44.png
try:
    _c31 = get_crop(31, 94, 43)
    canvas.paste(_c31, (20, 15), _c31)
except Exception:
    pass
layout["9.44"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/32_text_Events.png
try:
    _c32 = get_crop(32, 186, 56)
    canvas.paste(_c32, (46, 301), _c32)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/33_text_Thu_Apr_4_._5_00_PM_GMT_02_00.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 390), _c33)
except Exception:
    pass
layout["Thu,_Apr_4_._5:00_PM_GMT+"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/34_text_Al_Builders_Monthly_Networking_Food.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 390), _c34)
except Exception:
    pass
layout["Al_Builders_Monthly:_Netw"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/35_text_StartDock_Coworking_Prins_Hendrikkade.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 390), _c35)
except Exception:
    pass
layout["StartDock_Coworking_Prins"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/36_text_70_creator_followers.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 390), _c36)
except Exception:
    pass
layout["70_creator_followers"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/37_text_1_30AM_EDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1182), _c37)
except Exception:
    pass
layout["1:30AM_EDT"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/38_text_At_a_traffic_island_across_street_from.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 1182), _c38)
except Exception:
    pass
layout["At_a_traffic_island_acros"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/39_text_8_1030_creator_followers.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 1182), _c39)
except Exception:
    pass
layout["8_1030_creator_followers"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/40_text_Sat.png
try:
    _c40 = get_crop(40, 77, 45)
    canvas.paste(_c40, (390, 2030), _c40)
except Exception:
    pass
layout["Sat,"] = [390, 2030, 467, 2075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/41_text_1O_00_PM_EDT.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 1974), _c41)
except Exception:
    pass
layout["1O:00_PM_EDT"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/42_text_8_3375_creator_followers.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 1974), _c42)
except Exception:
    pass
layout["8_3375_creator_followers"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_08_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-10/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
