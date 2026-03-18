# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_09
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11.png
# step_index: 9/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Helper: hex to rgb
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# Canvas dimensions
W, H = canvas.size

# Colors (picked to match the screenshot tones)
COLOR_BG = _hex_to_rgb("#FFFFFF")            # page background (dominant)
COLOR_STATUS = _hex_to_rgb("#E6E7E8")        # top status bar light grey
HERO_TOP = _hex_to_rgb("#0b1b2b")            # hero image top tint (dark navy)
HERO_BOTTOM = _hex_to_rgb("#4b5f8a")         # hero image bottom tint (muted blue)
OVERLAY_TOOLBAR = _hex_to_rgb("#071221")     # toolbar overlay (darker)
CARD_BG = _hex_to_rgb("#F6F7F9")             # light card background
CARD_BORDER = _hex_to_rgb("#E1E3E8")         # subtle card border
TICKET_BORDER = _hex_to_rgb("#2F55FF")       # blue ticket border
SEPARATOR = _hex_to_rgb("#E9E9EA")           # section separator

# 1) Fill background (ensure canvas is set to the dominant color)
draw.rectangle([(0, 0), (W, H)], fill=COLOR_BG)

# 2) Status bar area at top (~50px)
status_h = 80  # slightly larger to accommodate various device status areas
draw.rectangle([(0, 0), (W, status_h)], fill=COLOR_STATUS)

# 3) Hero image background (dark gradient block). Keep it below status bar.
hero_top_y = status_h
hero_bottom_y = 560  # approximate height of the header image area in the screenshot

# Draw a vertical gradient for hero area
r1, g1, b1 = HERO_TOP
r2, g2, b2 = HERO_BOTTOM
height = max(1, hero_bottom_y - hero_top_y)
for i in range(height):
    t = i / float(height - 1)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    draw.line([(0, hero_top_y + i), (W, hero_top_y + i)], fill=(r, g, b))

# Subtle darker overlay near top of hero to anchor toolbar icons (do not draw icons themselves)
overlay_h = 120
draw.rectangle([(0, hero_top_y), (W, hero_top_y + overlay_h)], fill=OVERLAY_TOOLBAR)

# 4) Slight shadow divider under hero image
shadow_y = hero_bottom_y
draw.rectangle([(0, shadow_y), (W, shadow_y + 2)], fill=SEPARATOR)

# 5) Organizer profile card (rounded rectangle background behind organizer row)
# Keep this card clear of icons/text which will be pasted on top.
org_card_x0 = 40
org_card_x1 = W - 40
org_card_y0 = 1150
org_card_y1 = 1310
org_radius = 28

# Rounded rectangle (background)
try:
    draw.rounded_rectangle([(org_card_x0, org_card_y0), (org_card_x1, org_card_y1)],
                           radius=org_radius, fill=CARD_BG, outline=CARD_BORDER, width=1)
except Exception:
    # Fallback if rounded_rectangle not available
    draw.rectangle([(org_card_x0, org_card_y0), (org_card_x1, org_card_y1)], fill=CARD_BG, outline=CARD_BORDER)

# Tiny top/bottom divider lines to separate organizer card from surrounding content
draw.line([(org_card_x0 + 12, org_card_y1 + 18), (org_card_x1 - 12, org_card_y1 + 18)], fill=SEPARATOR, width=1)
draw.line([(org_card_x0 + 12, org_card_y0 - 18), (org_card_x1 - 12, org_card_y0 - 18)], fill=(255,255,255,0))

# 6) Section separators for content zones
# Under the main event details area (thin subtle line)
sep1_y = 1680
draw.line([(40, sep1_y), (W - 40, sep1_y)], fill=SEPARATOR, width=1)

# Another separator below the "About this event" text area
sep2_y = 2040
draw.line([(40, sep2_y), (W - 40, sep2_y)], fill=SEPARATOR, width=1)

# 7) Ticket selection card with blue outline (positioned above the checkout bar)
# Ensure the card does not overlap the detected checkout bar at y >= 2324
ticket_card_x0 = 40
ticket_card_x1 = W - 40
ticket_card_y1 = 2280  # bottom of ticket card (kept above checkout area)
ticket_card_height = 92
ticket_card_y0 = ticket_card_y1 - ticket_card_height
ticket_radius = 22

# Fill with white interior and blue border
try:
    draw.rounded_rectangle([(ticket_card_x0, ticket_card_y0), (ticket_card_x1, ticket_card_y1)],
                           radius=ticket_radius, fill=COLOR_BG, outline=TICKET_BORDER, width=8)
except Exception:
    draw.rectangle([(ticket_card_x0, ticket_card_y0), (ticket_card_x1, ticket_card_y1)], fill=COLOR_BG, outline=TICKET_BORDER)

# Inner subtle shadow for the ticket card (top)
draw.line([(ticket_card_x0 + 8, ticket_card_y0 + 6), (ticket_card_x1 - 8, ticket_card_y0 + 6)], fill=SEPARATOR)

# 8) Lightweight content area band behind event meta (location/time icons area)
meta_band_y0 = 1420
meta_band_y1 = 1760
# Keep it mostly background white but very faint tint to separate sections
meta_tint = (250, 250, 251)
draw.rectangle([(0, meta_band_y0), (W, meta_band_y1)], fill=meta_tint)

# 9) Small horizontal padding line under meta area
draw.line([(40, meta_band_y1), (W - 40, meta_band_y1)], fill=SEPARATOR, width=1)

# 10) Final top-of-footer separator (above the checkout bar area)
footer_sep_y = 2320
draw.line([(20, footer_sep_y), (W - 20, footer_sep_y)], fill=SEPARATOR, width=1)

# Note: All icons, buttons, and text are intentionally NOT drawn here.
# Structural backgrounds and separators have been placed to match the screenshot layout.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/01_icon_Check_out_for_S12.51.png
try:
    _c1 = get_crop(1, 1440, 636)
    canvas.paste(_c1, (0, 2324), _c1)
except Exception:
    pass
layout["Check_out_for_S12.51"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/03_icon_7.52.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["7.52"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 114, 107)
    canvas.paste(_c5, (987, 2439), _c5)
except Exception:
    pass
layout["icon_5"] = [987, 2439, 1101, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/06_icon_Music.png
try:
    _c6 = get_crop(6, 203, 101)
    canvas.paste(_c6, (41, 2070), _c6)
except Exception:
    pass
layout["Music"] = [41, 2070, 244, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 108, 105)
    canvas.paste(_c7, (1215, 2441), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 2441, 1323, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 92, 104)
    canvas.paste(_c8, (1108, 2441), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2441, 1200, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/09_icon_Ticket_sales_end_soon.png
try:
    _c9 = get_crop(9, 548, 85)
    canvas.paste(_c9, (40, 752), _c9)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 752, 588, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 98, 60)
    canvas.paste(_c10, (1216, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 2, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 57, 62)
    canvas.paste(_c11, (1316, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1316, 1, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/12_icon_7.52.png
try:
    _c12 = get_crop(12, 60, 68)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["7.52"] = [115, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/13_icon_7.52.png
try:
    _c13 = get_crop(13, 61, 67)
    canvas.paste(_c13, (181, 1), _c13)
except Exception:
    pass
layout["7.52"] = [181, 1, 242, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/14_icon_Jesse_LevIt_QUARTET.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1116, 108), _c14)
except Exception:
    pass
layout["Jesse_LevIt_QUARTET"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 65)
    canvas.paste(_c15, (248, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [248, 2, 301, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 63, 64)
    canvas.paste(_c16, (310, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [310, 2, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/17_icon_Escape_the_Routine.png
try:
    _c17 = get_crop(17, 414, 144)
    canvas.paste(_c17, (288, 1155), _c17)
except Exception:
    pass
layout["Escape_the_Routine"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/18_icon_S10.00.png
try:
    _c18 = get_crop(18, 106, 115)
    canvas.paste(_c18, (289, 2570), _c18)
except Exception:
    pass
layout["S10.00"] = [289, 2570, 395, 2685]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/19_text_7.52.png
try:
    _c19 = get_crop(19, 89, 43)
    canvas.paste(_c19, (22, 17), _c19)
except Exception:
    pass
layout["7.52"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/20_text_Wednesday_April_24.png
try:
    _c20 = get_crop(20, 414, 144)
    canvas.paste(_c20, (288, 1155), _c20)
except Exception:
    pass
layout["Wednesday;_April_24"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/21_text_7_00_PM.png
try:
    _c21 = get_crop(21, 207, 56)
    canvas.paste(_c21, (585, 893), _c21)
except Exception:
    pass
layout["7:00_PM"] = [585, 893, 792, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/22_text_Art_Rangers_Jazz_Night_Auction.png
try:
    _c22 = get_crop(22, 414, 144)
    canvas.paste(_c22, (288, 1155), _c22)
except Exception:
    pass
layout["Art_Rangers_Jazz_Night_&_"] = [288, 1155, 702, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/23_text_The_Faight_Collective.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1422), _c23)
except Exception:
    pass
layout["The_Faight_Collective"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/24_text_3_hrs.png
try:
    _c24 = get_crop(24, 112, 49)
    canvas.paste(_c24, (141, 1580), _c24)
except Exception:
    pass
layout["3_hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1685), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1422), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/27_text_About_this_event.png
try:
    _c27 = get_crop(27, 455, 65)
    canvas.paste(_c27, (44, 1982), _c27)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 499, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/28_text_Join_us_at_The_Faight_Collective_for_the.png
try:
    _c28 = get_crop(28, 1440, 636)
    canvas.paste(_c28, (0, 2324), _c28)
except Exception:
    pass
layout["Join_us_at_The_Faight_Col"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/29_text_General_Admission.png
try:
    _c29 = get_crop(29, 415, 55)
    canvas.paste(_c29, (116, 2451), _c29)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/30_text_S10.00.png
try:
    _c30 = get_crop(30, 163, 57)
    canvas.paste(_c30, (113, 2592), _c30)
except Exception:
    pass
layout["S10.00"] = [113, 2592, 276, 2649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_09_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-11/31_clickable_Organizer_profile_picture.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (96, 1194), _c31)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1194, 240, 1338]
