
#' @param se1e2iiar_2strains_pop Data frame containing `location_code`, `S`, `E1`, `E2`, `I`, `Ia`, `R`, `E1_b`, `E2_b`, `I_b`, and`Ia_b` for the entire population
#' @param mobility_matrix List (1 entry for each time period) with each entry containing a data frame with `from`, `to`, `n` for the number of people who travel. Each data frame must be complete.
#' @param dynamic_seeds Data.table containing `location_code`, `day`, `n`, `n_b` for dynamic seeding (or `NULL`)
#' @param betas Data table with `locaton code`, `day`, `time`, `beta` (1 entry for each location for each time period). Float, infection parameter, 0.6
#' @param inputSeed Integer with seed for cpp code
#' @param latent_period Float, 3.0
#' @param presymptomatic_period Float 2.0
#' @param infectious_period Float, 5.0
#' @param b_relative_infectiousness Float, relative infectiousness of strain B
#' @param presymptomatic_relative_infectiousness Float, relative infectiousness of presymptomatic
#' @param asymptomatic_prob Float, Proportion/probability of asymptomatic given infectious
#' @param asymptomatic_relative_infectiousness Float, Relative infectiousness of asymptomatic infectious
#' @param N Int = 1 int, Number of internal simulations (average taken). This is generally used for parameter fitting.

#' \item{location_code}{Location code}
#' \item{week}{Week number}
#' \item{day}{Day number}
#' \item{time}{Time of reporting (23:59)}
#' \item{b_S}{Susceptibles belonging to location code}
#' \item{b_E1}{E1s belonging to location code}
#' \item{b_E2}{E2s belonging to location code}
#' \item{b_I}{Infectious (symptomatic) belonging to location code}
#' \item{b_Ia}{Infectious (asymptomatic) belonging to location code}
#' \item{b_R}{Recovered belonging to location code}
#' \item{b_E1_b}{E1_bs belonging to location code}
#' \item{b_E2_b}{E2_bs belonging to location code}
#' \item{b_I_b}{Infectious (symptomatic) strain B belonging to location code}
#' \item{b_Ia_b}{Infectious (asymptomatic) strain B belonging to location code}
#' \item{c_S}{Susceptibles currently in this location code}
#' \item{c_E1}{E1s currently in this location code}
#' \item{c_E2}{E2s currently in this location code}
#' \item{c_I}{Infectious (symptomatic) currently in this location code}
#' \item{c_Ia}{Infectious (asymptomatic) currently in this location code}
#' \item{c_R}{Recovered currently in this location code}
#' \item{c_E1_b}{E1_bs currently in this location code}
#' \item{c_E2_b}{E2_bs currently in this location code}
#' \item{c_I_b}{Infectious (symptomatic) strain B currently in this location code}
#' \item{c_Ia_b}{Infectious (asymptomatic) strain B currently in this location code}
#' \item{c_symp_incidence_a}{Transition from E2 to I currently in this location code}
#' \item{c_asymp_incidence_a}{Transition from E1 to Ia currently in this location code}
#' \item{c_symp_incidence_b}{Transition from E2_b to I_b currently in this location code}
#' \item{c_asymp_incidence_b}{Transition from E1_b to Ia_b currently in this location code}
#' \item{c_symp_incidence}{Transition from E2 and E2_b to I and I_b currently in this location code}
#' \item{c_asymp_incidence}{Transition from E1 and E1_b to Ia and Ia_b currently in this location code}

#' @examples
#' spread::asymmetric_mobility_se1e2iiar_2strains(
#'   se1e2iiar_2strains_pop = spread::asymmetric_mobility_se1e2iiar_dummy_se1e2iiar_2strains_pop,
#'   mobility_matrix = spread::asymmetric_mobility_se1e2iiar_dummy_mobility_matrix,
#'   dynamic_seeds = spread::asymmetric_mobility_se1e2iiar_2strains_dummy_dynamic_seeds,
#'   betas = spread::asymmetric_mobility_se1e2iiar_dummy_betas,
#'   inputSeed = 123,
#'   latent_period = 3.0,
#'   presymptomatic_period = 2.0,
#'   infectious_period = 5.0,
#'   b_relative_infectiousness = 1.5,
#'   presymptomatic_relative_infectiousness = 1,
#'   asymptomatic_prob = 0,
#'   asymptomatic_relative_infectiousness = 0,
#'   N = 1
#' )




def asymmetric_mobility_se1e2iiar(se1e2iiar_pop, mobility_matrix, dynamic_seeds, betas, inputSeed, 
                                latent_period, presymptomatic_period, infectious_period, presymptomatic_relative_infectiousness, 
                                asymptomatic_prob, asymptomatic_relative_infectiousness, N):
    
    # Parameters used in epidemiological model
    a1 = 1 / latent_period             #  Finn ut av dette
    a2 = 1 / presymptomatic_period     #  Finn ut av dette
    gamma = 1 / infectious_period      #  Finn ut av dette

    # create seed_matrix from dynamic_seeds
    location_codes = ...
    
    # location_codes <- se1e2iiar_2strains_pop$location_code
    # if(!is.null(dynamic_seeds)){
    #     dynamic_seeds_a <- data.table::copy(dynamic_seeds)[,"n_b":=NULL]
    #     dynamic_seeds_b <- data.table::copy(dynamic_seeds)[,"n":=NULL]
    #     names(dynamic_seeds_b)[names(dynamic_seeds_b) == "n_b"] = "n"
    # }
    # else{
    #     dynamic_seeds_a = NULL
    #     dynamic_seeds_b = NULL
    # }
    # seed_matrix <- convert_dynamic_seeds_to_seed_matrix(
    #     dynamic_seeds = dynamic_seeds_a,
    #     location_codes = location_codes,
    #     days = 1:days_simulation
    # )

    # seed_matrix_b <- convert_dynamic_seeds_to_seed_matrix(
    #     dynamic_seeds = dynamic_seeds_b,
    #     location_codes = location_codes,
    #     days = 1:days_simulation
    # )

    # # create beta_matrix from betas data frame
    # location_codes <- se1e2iiar_2strains_pop$location_code
    # beta_matrix <- convert_beta_to_matrix(
    #     betas = betas,
    #     location_codes = location_codes,
    #     days = 1:days_simulation,
    #     times = c(0, 6, 12, 18)
    # )




def main():
    se1e2iiar_pop = ...
    mobility_matrix = ...
    dynamic_seeds = ...
    betas = ...
    inputSeed = ...
    latent_period = 3.0
    presymptomatic_period = 2.0
    infectious_period = 5.0
    presymptomatic_relative_infectiousness = 1.25
    asymptomatic_prob = 0.4
    asymptomatic_relative_infectiousness = 0.5
    N = 1
    betas = betas 

    asymmetric_mobility_se1e2iiar(se1e2iiar_pop, mobility_matrix, dynamic_seeds, betas, inputSeed, 
                                    latent_period, presymptomatic_period, infectious_period, presymptomatic_relative_infectiousness, 
                                    asymptomatic_prob, asymptomatic_relative_infectiousness, N)
    








def rda_to_csv(rda):
    
    return csv