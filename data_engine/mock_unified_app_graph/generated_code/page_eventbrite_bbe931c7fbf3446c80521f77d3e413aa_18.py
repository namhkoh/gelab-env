# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_18
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20.png
# step_index: 18/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for 1440x2960 canvas using provided `canvas` and `draw`.

# Colors
bg_color = "#fafafa"         # page background
status_bar_color = "#d0d0d0" # top status bar
header_bg = "#ffffff"        # header/toolbar background
header_border = "#e6e6ea"
card_fill_dark = "#2f3542"   # dark media card background
card_fill_accent = "#12a3d6" # secondary media card accent
card_border = "#e6e6ea"
ticket_border = "#4a3fb8"    # ticket card border (purple tone)
ticket_bg = "#ffffff"
reserve_btn = "#c84d20"      # orange reserve button
separator = "#ececf2"

w, h = canvas.size

# Page background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Thin subtle divider under status bar
draw.line([(0, status_h), (w, status_h)], fill="#cfcfcf", width=1)

# Header / toolbar area (rounded white container under status bar)
header_top = status_h + 8
header_bottom = header_top + 82
draw.rounded_rectangle(
    [(20, header_top), (w - 20, header_bottom)],
    radius=18,
    fill=header_bg,
    outline=header_border,
    width=2
)

# Subtle shadow under header (simulated with a faint line)
draw.line([(24, header_bottom + 2), (w - 24, header_bottom + 2)], fill="#f0f0f4", width=1)

# Main content area separators
content_left = 36
content_right = w - 36

# First large media card (rounded) - background behind first hero image/video
media1_top = 360
media1_bottom = 760
draw.rounded_rectangle(
    [(content_left, media1_top), (content_right, media1_bottom)],
    radius=34,
    fill=card_fill_dark,
    outline=card_border,
    width=2
)

# Subtle progress bar background at bottom of media1 (low-contrast, decorative)
pb_h = 12
pb_margin = 60
draw.rectangle(
    [(content_left + pb_margin, media1_bottom - 26), (content_right - pb_margin, media1_bottom - 26 + pb_h)],
    fill="#dcdfe6"
)

# Second media/video card area (YouTube style) - a bright accent background to sit behind embedded player
media2_top = media1_bottom + 40
media2_bottom = media2_top + 520
draw.rounded_rectangle(
    [(content_left, media2_top), (content_right, media2_bottom)],
    radius=18,
    fill=card_fill_accent,
    outline=card_border,
    width=2
)

# Add a darker strip near top of media2 to suggest title bar area (purely structural)
draw.rectangle(
    [(content_left + 12, media2_top + 12), (content_right - 12, media2_top + 64)],
    fill="#0f7fa0"
)

# Thin separator line between media2 and following content
sep_y = media2_bottom + 18
draw.line([(content_left, sep_y), (content_right, sep_y)], fill=separator, width=2)

# Content strip below media (muted preview area)
preview_top = sep_y + 18
preview_bottom = preview_top + 220
draw.rounded_rectangle(
    [(content_left, preview_top), (content_right, preview_bottom)],
    radius=12,
    fill="#f5f6f8",
    outline="#e9e9ee",
    width=1
)

# Ticket selection card near bottom (Complimentary Access)
ticket_top = 2320
ticket_bottom = 2620
draw.rounded_rectangle(
    [(24, ticket_top), (w - 24, ticket_bottom)],
    radius=22,
    fill=ticket_bg,
    outline=ticket_border,
    width=6
)

# Internal subtle divider inside ticket card
draw.line([(48, ticket_top + 84), (w - 48, ticket_top + 84)], fill=separator, width=1)

# Small light rounded container for quantity/controls on right side of ticket card (structural only)
qty_w = 170
qty_h = 84
qty_right = w - 72
qty_left = qty_right - qty_w
qty_top = ticket_top + 28
qty_bottom = qty_top + qty_h
draw.rounded_rectangle(
    [(qty_left, qty_top), (qty_right, qty_bottom)],
    radius=12,
    fill="#f6f7fb",
    outline="#e6e6ee",
    width=2
)

# Reserve button at bottom (prominent orange CTA)
reserve_top = 2740
reserve_bottom = reserve_top + 120
btn_margin = 60
draw.rounded_rectangle(
    [(btn_margin, reserve_top), (w - btn_margin, reserve_bottom)],
    radius=14,
    fill=reserve_btn,
    outline=None,
    width=0
)

# Bottom safe area subtle line
draw.line([(0, reserve_bottom + 6), (w, reserve_bottom + 6)], fill="#f3f3f5", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/00_icon_Franchising_events_in.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["Franchising_events_in"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/04_icon_Increase.png
try:
    _c4 = get_crop(4, 96, 96)
    canvas.paste(_c4, (1224, 2444), _c4)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 92, 103)
    canvas.paste(_c5, (1108, 2441), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/06_icon_Inside_the_Franchise_Expo_West.png
try:
    _c6 = get_crop(6, 1323, 755)
    canvas.paste(_c6, (58, 1270), _c6)
except Exception:
    pass
layout["(Inside_the_Franchise_Exp"] = [58, 1270, 1381, 2025]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 50, 54)
    canvas.paste(_c7, (316, 8), _c7)
except Exception:
    pass
layout["icon_7"] = [316, 8, 366, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/08_icon_9.13.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 54)
    canvas.paste(_c9, (250, 7), _c9)
except Exception:
    pass
layout["icon_9"] = [250, 7, 300, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 55)
    canvas.paste(_c10, (184, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [184, 5, 235, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 44, 59)
    canvas.paste(_c11, (1157, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [1157, 5, 1201, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/12_icon_9.13.png
try:
    _c12 = get_crop(12, 52, 55)
    canvas.paste(_c12, (117, 6), _c12)
except Exception:
    pass
layout["9.13"] = [117, 6, 169, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 56)
    canvas.paste(_c13, (1326, 6), _c13)
except Exception:
    pass
layout["icon_13"] = [1326, 6, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 81, 57)
    canvas.paste(_c14, (1214, 5), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 5, 1295, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 46, 54)
    canvas.paste(_c15, (384, 7), _c15)
except Exception:
    pass
layout["icon_15"] = [384, 7, 430, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/16_icon_THELIFE-CHANGING.png
try:
    _c16 = get_crop(16, 1323, 755)
    canvas.paste(_c16, (58, 1270), _c16)
except Exception:
    pass
layout["THELIFE-CHANGING"] = [58, 1270, 1381, 2025]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/17_icon_Free.png
try:
    _c17 = get_crop(17, 133, 122)
    canvas.paste(_c17, (99, 2567), _c17)
except Exception:
    pass
layout["Free"] = [99, 2567, 232, 2689]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/18_icon_franchising_recap.png
try:
    _c18 = get_crop(18, 1127, 33)
    canvas.paste(_c18, (144, 2075), _c18)
except Exception:
    pass
layout["franchising_recap"] = [144, 2075, 1271, 2108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/19_icon_Free.png
try:
    _c19 = get_crop(19, 75, 72)
    canvas.paste(_c19, (249, 2588), _c19)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/20_icon_Share.png
try:
    _c20 = get_crop(20, 65, 84)
    canvas.paste(_c20, (1285, 1270), _c20)
except Exception:
    pass
layout["Share"] = [1285, 1270, 1350, 1354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/21_icon_Reserve_a_spot.png
try:
    _c21 = get_crop(21, 99, 96)
    canvas.paste(_c21, (996, 2444), _c21)
except Exception:
    pass
layout["Reserve_a_spot"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 43, 56)
    canvas.paste(_c22, (1271, 6), _c22)
except Exception:
    pass
layout["icon_22"] = [1271, 6, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/23_text_9.13.png
try:
    _c23 = get_crop(23, 91, 43)
    canvas.paste(_c23, (20, 17), _c23)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/24_text_Minorities_Building_..png
try:
    _c24 = get_crop(24, 560, 87)
    canvas.paste(_c24, (249, 146), _c24)
except Exception:
    pass
layout["Minorities_Building_."] = [249, 146, 809, 233]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/25_text_REGISTER_TODAY_TO_ALSO_GET_FULL_ACCESS_T.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (36, 108), _c25)
except Exception:
    pass
layout["REGISTER_TODAY_TO_ALSO_GE"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/26_text_FRANCHISE_EXPO_WEST_EXHIBIT_HALL..png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (36, 108), _c26)
except Exception:
    pass
layout["FRANCHISE_EXPO_WEST_EXHIB"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/27_text_Here_is_a_recap_of_our_last_15_Minoritie.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (36, 108), _c27)
except Exception:
    pass
layout["Here_is_a_recap_of_our_la"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/28_text_Los_Angeles_Phoenix_x_2_and_Long_Beach.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (36, 108), _c28)
except Exception:
    pass
layout["Los_Angeles,_Phoenix_x_2,"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/29_clickable_Photo_image_of_Sociallybuzz_Inc.png
try:
    _c29 = get_crop(29, 66, 66)
    canvas.paste(_c29, (68, 1280), _c29)
except Exception:
    pass
layout["Photo_image_of_Sociallybu"] = [68, 1280, 134, 1346]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/30_clickable_From_Franchise_Workshop_Attendee_to_Fran.png
try:
    _c30 = get_crop(30, 1127, 33)
    canvas.paste(_c30, (144, 1299), _c30)
except Exception:
    pass
layout["From_Franchise_Workshop_A"] = [144, 1299, 1271, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/31_clickable_Play.png
try:
    _c31 = get_crop(31, 93, 66)
    canvas.paste(_c31, (673, 1615), _c31)
except Exception:
    pass
layout["Play"] = [673, 1615, 766, 1681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/32_clickable_Watch_on_YouTube.png
try:
    _c32 = get_crop(32, 238, 65)
    canvas.paste(_c32, (58, 1953), _c32)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 1953, 296, 2018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/33_clickable_Share.png
try:
    _c33 = get_crop(33, 65, 84)
    canvas.paste(_c33, (1285, 2046), _c33)
except Exception:
    pass
layout["Share"] = [1285, 2046, 1350, 2130]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_18_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-20/34_clickable_Photo_image_of_Sociallybuzz_Inc.png
try:
    _c34 = get_crop(34, 66, 66)
    canvas.paste(_c34, (68, 2056), _c34)
except Exception:
    pass
layout["Photo_image_of_Sociallybu"] = [68, 2056, 134, 2122]
