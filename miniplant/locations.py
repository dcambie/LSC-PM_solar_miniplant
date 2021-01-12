from pvlib.location import Location

TOWNSVILLE = Location(latitude=-19.3239872, longitude=146.7605092, tz='Australia/Brisbane', altitude=16.3, name='Townsville')
PLATAFORMA_SOLAR_ALMERIA = Location(latitude=37.09454882268096, longitude=-2.3586145374427008, tz='Europe/Madrid', altitude=499, name='Plataforma Solar de Almería')
EINDHOVEN = Location(latitude=51.4416, longitude=5.6497, tz='Europe/Amsterdam', altitude=17, name='Eindhoven')
NORTH_CAPE = Location(latitude=70.976021, longitude=25.983061, tz='Europe/Stockholm', altitude=17, name='North Cape')

LOCATIONS = {TOWNSVILLE, PLATAFORMA_SOLAR_ALMERIA, EINDHOVEN, NORTH_CAPE}
