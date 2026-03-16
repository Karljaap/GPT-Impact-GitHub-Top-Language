
//----------------------------------------------------------------------------//
//
// Proyecto: Tesis
// Impacto de ChatGPT en el número de programadores en GitHub
//
//----------------------------------------------------------------------------//

global path "C:/Users/ronco/Desktop/GPT-Impact-GitHub-Top-Language"

* Crear carpetas si no existen
cap mkdir "$path/output"
cap mkdir "$path/output/figures"
cap mkdir "$path/output/tables"

import delimited "$path/output/data/data_langs_balanced.csv", clear

sort unique_id year quarter
drop if iso2_code == "HK"
label var num_pushers_pc "Número de pushers por 100k habitantes"
label var gpt_available_post1 "ChatGPT Disponible"

**************************
// Cambio de nombres de etiquetas
**************************
replace language = "C_hashtag" if language == "C#"
replace language = "C_plus"    if language == "C++"

*****************
// Tesis_DataScience
*****************
local DataScience "C C_hashtag C_plus Go Java JavaScript PHP Python Ruby TypeScript"

foreach lang of local DataScience {

        local l`v' : variable label num_pushers_pc

        *-----------------------------------------------------
        * DID
        *-----------------------------------------------------
        eststo `lang'_did: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(did) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'did", .png)

        scalar b_`lang'_did  = _b[gpt_available_post1]
        scalar se_`lang'_did = _se[gpt_available_post1]
        scalar nobs_`lang'   = e(N)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'did_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'did_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'

        *-----------------------------------------------------
        * SC
        *-----------------------------------------------------
        eststo `lang'_sc: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(sc) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'sc", .png)

        scalar b_`lang'_sc  = _b[gpt_available_post1]
        scalar se_`lang'_sc = _se[gpt_available_post1]

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'sc_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'sc_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'

        *-----------------------------------------------------
        * SDID
        *-----------------------------------------------------
        eststo `lang'_sdid: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(sdid) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'sdid", .png)

        scalar b_`lang'_sdid  = _b[gpt_available_post1]
        scalar se_`lang'_sdid = _se[gpt_available_post1]

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'sdid_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'sdid_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        scalar cmean_`lang' = r(mean)
        estadd scalar control_mean `r(mean)'
}

** ── Construir tabla LaTeX ────────────────────────────────────────────────────

file open fh using "$path/output/tables/gpt_impact_github_DataScience.tex", write replace

file write fh "\begin{table}[htbp]\centering" _n
file write fh "\caption{Impacto de ChatGPT en el n\'{u}mero de programadores}" _n
file write fh "{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}" _n
file write fh "\begin{tabular}{lccccc}" _n
file write fh "\toprule" _n
file write fh "Lenguaje & DID & SC & SDID & Obs. & \shortstack{Baseline \\\\ Mean} \\\\" _n
file write fh "\midrule" _n

foreach lang of local DataScience {
    if "`lang'" == "C"          local label "C"
    if "`lang'" == "C_hashtag"  local label "C\#"
    if "`lang'" == "C_plus"     local label "C++"
    if "`lang'" == "Go"         local label "Go"
    if "`lang'" == "Java"       local label "Java"
    if "`lang'" == "JavaScript" local label "JavaScript"
    if "`lang'" == "PHP"        local label "PHP"
    if "`lang'" == "Python"     local label "Python"
    if "`lang'" == "Ruby"       local label "Ruby"
    if "`lang'" == "TypeScript" local label "TypeScript"

    local t1 = scalar(b_`lang'_did)  / scalar(se_`lang'_did)
    local t2 = scalar(b_`lang'_sc)   / scalar(se_`lang'_sc)
    local t3 = scalar(b_`lang'_sdid) / scalar(se_`lang'_sdid)

    local p1 = 2*(1-normal(abs(`t1')))
    local p2 = 2*(1-normal(abs(`t2')))
    local p3 = 2*(1-normal(abs(`t3')))

    if `p1' < 0.01      local s1 "***"
    else if `p1' < 0.05 local s1 "**"
    else if `p1' < 0.10 local s1 "*"
    else                local s1 ""

    if `p2' < 0.01      local s2 "***"
    else if `p2' < 0.05 local s2 "**"
    else if `p2' < 0.10 local s2 "*"
    else                local s2 ""

    if `p3' < 0.01      local s3 "***"
    else if `p3' < 0.05 local s3 "**"
    else if `p3' < 0.10 local s3 "*"
    else                local s3 ""

    local B1 = string(scalar(b_`lang'_did),   "%9.3f")
    local B2 = string(scalar(b_`lang'_sc),    "%9.3f")
    local B3 = string(scalar(b_`lang'_sdid),  "%9.3f")
    local E1 = string(scalar(se_`lang'_did),  "%9.3f")
    local E2 = string(scalar(se_`lang'_sc),   "%9.3f")
    local E3 = string(scalar(se_`lang'_sdid), "%9.3f")
    local N  = string(scalar(nobs_`lang'),     "%9.0f")
    local CM = string(scalar(cmean_`lang'),    "%9.3f")

    file write fh "`label' & `B1'`s1' & `B2'`s2' & `B3'`s3' & `N' & `CM' \\" _n
    file write fh "              & (`E1') & (`E2') & (`E3') & & \\" _n
    file write fh "\addlinespace" _n
}

file write fh "\bottomrule" _n
file write fh "\end{tabular}}" _n
file write fh "\begin{minipage}{\linewidth}" _n
file write fh "\footnotesize \textit{Nota.} Errores est\'{a}ndar entre par\'{e}ntesis. * p<0.10, ** p<0.05, *** p<0.01" _n
file write fh "\end{minipage}" _n
file write fh "\end{table}" _n

file close fh

cd "$path"
