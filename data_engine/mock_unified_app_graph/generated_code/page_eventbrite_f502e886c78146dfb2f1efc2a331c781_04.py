# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_04
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6.png
# step_index: 4/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 canvas.
# Variables provided: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall base background (match screenshot dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top ~64px) - light gray background to emulate system status bar
status_h = 64
draw.rectangle([(0, 0), (1440, status_h)], fill="#D3D3D3")

# Subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#BFBFBF", width=1)

# Header / Search area
header_top = status_h
header_h = 116
header_rect = [(0, header_top), (1440, header_top + header_h)]
# keep header white but give a very faint off-white fill to separate from base
draw.rectangle(header_rect, fill="#FFFFFF")

# Blue underline for the search field (prominent accent)
underline_y = header_top + header_h - 6
draw.line([(48, underline_y), (1440-48, underline_y)], fill="#2E56F0", width=6)

# subtle shadow under header
draw.line([(0, header_top + header_h), (1440, header_top + header_h)], fill="#E8E8E8", width=1)

# Rounded background behind the search area (subtle, does not draw any icons/text)
search_bg = [(36, header_top + 20), (1440 - 36, header_top + header_h - 18)]
draw.rounded_rectangle(search_bg, radius=12, fill="#F7F9FF")

# Main content area separators and section grouping
# Event rows top positions (from detected crops)
event_tops = [390, 786, 1182, 1578, 1974, 2370]

# Draw thin separators between list items
sep_color = "#EFEFF1"
for y in [350] + event_tops + [2804]:
    draw.line([(36, y), (1440-36, y)], fill=sep_color, width=1)

# Draw subtle card-like rounded backgrounds behind each event row (keeps them distinct from page bg)
# These are just background panels — no text or icons are drawn.
card_x1 = 36
card_x2 = 1440 - 36
card_radius = 10
card_pad_v = 8
card_fill = "#FFFFFF"  # maintain white cards on white page but add a faint shadow outline
card_outline = "#F0F0F2"
for top in event_tops:
    y1 = top - card_pad_v
    y2 = top + 396 - 8  # approximate item height minus a bit
    draw.rounded_rectangle([(card_x1, y1), (card_x2, y2)], radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# Accent colored side strip for featured area (left margin band under header) - unobtrusive background element
# This sits behind the items and does not overlap detected icons/text in a way that duplicates them.
draw.rectangle([(24, header_top + header_h + 8), (36, 2800)], fill="#F7F8FF")

# Subtle alternating tint behind every other card (very light) to aid visual grouping
alt_fill = "#FEFEFF"
for i, top in enumerate(event_tops):
    if i % 2 == 1:
        y1 = top - card_pad_v
        y2 = top + 396 - 8
        draw.rectangle([(card_x1+2, y1+2), (card_x2-2, y2-2)], fill=alt_fill)

# Bottom navigation bar background (sticky footer)
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill="#FFFFFF")
# top border for the nav
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill="#E6E6E8", width=1)

# Small indicator strip above bottom nav for separation
draw.rectangle([(0, bottom_nav_top-6), (1440, bottom_nav_top-4)], fill="#F4F4F6")

# Left edge vertical protective margin (visual subtle guide)
draw.line([(36, header_top + header_h + 8), (36, bottom_nav_top - 12)], fill="#FAFAFB", width=1)

# Final subtle vignette/shadow at page edges (very light) to match screenshot feel
edge_shadow_color = "#FBFBFB"
# top inner shadow
draw.rectangle([(0, header_top + header_h), (1440, header_top + header_h + 2)], fill=edge_shadow_color)
# bottom inner shadow above nav
draw.rectangle([(0, bottom_nav_top - 2), (1440, bottom_nav_top)], fill=edge_shadow_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/00_icon_Event_s_image.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1974), _c0)
except Exception:
    pass
layout["Event's_image"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/01_icon_Renbel.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 390), _c1)
except Exception:
    pass
layout["Renbel"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/02_icon_BROTH_ERSc.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1182), _c2)
except Exception:
    pass
layout["BROTH_ERSc"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/03_icon_Festival.png
try:
    _c3 = get_crop(3, 1344, 191)
    canvas.paste(_c3, (48, 72), _c3)
except Exception:
    pass
layout["Festival"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/04_icon_alenor.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 2370), _c4)
except Exception:
    pass
layout["alenor_|"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/05_icon_Music.png
try:
    _c5 = get_crop(5, 54, 61)
    canvas.paste(_c5, (314, 3), _c5)
except Exception:
    pass
layout["Music"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/06_icon_Crawl.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 786), _c6)
except Exception:
    pass
layout["Crawl"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/07_icon_8_63_creator_followers.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1974), _c7)
except Exception:
    pass
layout["8_63_creator_followers"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/08_icon_7.18.png
try:
    _c8 = get_crop(8, 54, 63)
    canvas.paste(_c8, (183, 2), _c8)
except Exception:
    pass
layout["7.18"] = [183, 2, 237, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/09_icon_7.18.png
try:
    _c9 = get_crop(9, 59, 64)
    canvas.paste(_c9, (114, 1), _c9)
except Exception:
    pass
layout["7.18"] = [114, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/10_icon_Music.png
try:
    _c10 = get_crop(10, 46, 59)
    canvas.paste(_c10, (251, 4), _c10)
except Exception:
    pass
layout["Music"] = [251, 4, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/11_icon_Brews_Brothers_Brewpub.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1182), _c11)
except Exception:
    pass
layout["Brews_Brothers_Brewpub"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/12_icon_International_LA_Punk_Film_and_Music.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1578), _c12)
except Exception:
    pass
layout["International_LA_Punk_Fil"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/13_icon_8_125_creator_followers.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 786), _c13)
except Exception:
    pass
layout["8_125_creator_followers"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/14_icon_7.18.png
try:
    _c14 = get_crop(14, 113, 104)
    canvas.paste(_c14, (59, 118), _c14)
except Exception:
    pass
layout["7.18"] = [59, 118, 172, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/15_icon_enjoyment_are_extremely_attractive.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 2370), _c15)
except Exception:
    pass
layout["enjoyment_are_extremely_a"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 59, 64)
    canvas.paste(_c16, (1216, 0), _c16)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1275, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 53, 65)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["Cancel"] = [1318, 0, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 41, 65)
    canvas.paste(_c18, (1272, 0), _c18)
except Exception:
    pass
layout["Cancel"] = [1272, 0, 1313, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 149, 144)
    canvas.paste(_c19, (1243, 97), _c19)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1099, 96), _c20)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/21_icon_6167_Bristol_Parkway_Culver.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["6167_Bristol_Parkway;_Cul"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/22_icon_MAY_4TH2024.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 1578), _c22)
except Exception:
    pass
layout["MAY?4TH2024"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/23_icon_Drifting_On_A_Memory_Music_Festival.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1974), _c23)
except Exception:
    pass
layout["Drifting_On_A_Memory_Musi"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/24_icon_Greenbelt_Music_Festival.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 390), _c24)
except Exception:
    pass
layout["Greenbelt_Music_Festival"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 46, 61)
    canvas.paste(_c25, (384, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [384, 3, 430, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/27_icon_6167_Bristol_Parkway_Culver.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["6167_Bristol_Parkway;_Cul"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/28_icon_Greenbelt_Music_Festival.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 390), _c28)
except Exception:
    pass
layout["Greenbelt_Music_Festival"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/29_icon_International_LA_Punk_Film_and_Music.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1578), _c29)
except Exception:
    pass
layout["International_LA_Punk_Fil"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/30_icon_May.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 390), _c30)
except Exception:
    pass
layout["May"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/31_icon_The_Eclectic.png
try:
    _c31 = get_crop(31, 204, 52)
    canvas.paste(_c31, (390, 1023), _c31)
except Exception:
    pass
layout["The_Eclectic"] = [390, 1023, 594, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/32_icon_7.18.png
try:
    _c32 = get_crop(32, 95, 62)
    canvas.paste(_c32, (14, 2), _c32)
except Exception:
    pass
layout["7.18"] = [14, 2, 109, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/33_icon_More.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (1152, 2804), _c33)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/34_text_Events.png
try:
    _c34 = get_crop(34, 186, 56)
    canvas.paste(_c34, (46, 301), _c34)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_04_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-6/35_clickable_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]
