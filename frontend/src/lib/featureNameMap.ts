const featureNameMap: Record<string, string> = {
  etch_rate_std: 'Etch Rate Std Dev',
  etch_rate_mean: 'Etch Rate Mean',
  temperature_c: 'Chamber Temperature (°C)',
  pressure_mbar: 'Chamber Pressure (mbar)',
  flow_slm: 'Gas Flow (SLM)',
  rf_power_w: 'RF Power (W)',
  deposition_rate: 'Deposition Rate',
  uniformity_pct: 'Uniformity (%)',
  thickness_nm: 'Film Thickness (nm)',
  cd_nm: 'Critical Dimension (nm)',
  overlay_nm: 'Overlay Error (nm)',
  dose_cm2: 'Implant Dose (cm⁻²)',
  energy_kev: 'Implant Energy (keV)',
  removal_rate_nm_min: 'CMP Removal Rate',
  planarity_nm: 'Post-CMP Planarity (nm)',
  sheet_res_ohm_sq: 'Sheet Resistance (Ω/□)',
  contact_angle_deg: 'Contact Angle (°)',
  particle_count: 'Particle Count',
  reflectivity_pct: 'Reflectivity (%)',
  focus_offset_nm: 'Focus Offset (nm)',
}

export function humanize(raw: string): string {
  if (featureNameMap[raw]) return featureNameMap[raw]
  return raw.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export default featureNameMap
